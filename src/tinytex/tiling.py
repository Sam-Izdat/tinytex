import typing
from typing import Union, Tuple

import torch

from scipy.sparse import lil_matrix
import scipy.signal

from .geometry import Geometry

lss_enabled = "spsolve"
try:
    import pyamgcl as lss
    lss_enabled = "amgcl"
except ImportError:
    try:
        import pyamg as lss  
        lss_enabled = "pyamg"
    except ImportError:
        from scipy.sparse.linalg import spsolve as lss

from .util import *

class Tiling:

    """Tiling and tile blending."""

    @classmethod
    def get_tile_position(cls, idx:int, cols:int) -> (int, int):
        """
        Get row and column position of tile in tile grid, by tile index.

        :param idx: tile index
        :param cols: total number of columns in tile grid
        :return: row and column position, respectively
        """
        row = idx // cols
        col = idx % cols
        return row, col

    @classmethod
    def get_tile_index(cls, r:int, c:int, cols:int) -> int:
        """
        Get tile index, by row and column position.

        :param idx: Tile index.
        :param r: Tile's row position.
        :param c: Tile's column position.
        :param cols: Total number of columns in tile grid.
        :return: row and column position, respectively
        """
        return (r * cols) + c

    @classmethod
    def get_tile_neighbors(cls, r:int, c:int, rows:int, cols:int, wrap:bool=False) -> (int, int, int, int):
        """
        Get indices of adjacent tiles from tile's row and column position.

        :param r: Tile's row position.
        :param c: Tile's column position.
        :param rows: Total number of rows in tile grid.
        :param cols: Total number of columns in tile grid.
        :param wrap: Wrap around edge tiles to opposite side of grid.
        :return: Tile indices of top, right, bottom, and left neighboring tiles, respectively, 
            or -1 if no neighboring tile (when wrap is False).
        """
        top =       (r-1)*cols+c if r > 0       else (rows-1)*cols+c        if wrap else -1
        right =     r*cols+(c+1) if c < cols-1  else r*cols                 if wrap else -1
        bottom =    (r+1)*cols+c if r < rows-1  else c                      if wrap else -1
        left =      r*cols+(c-1) if c > 0       else r*cols+(cols-1)        if wrap else -1
        
        return top, right, bottom, left

    @classmethod
    def split(cls, im:torch.Tensor, shape:tuple) -> (torch.Tensor, int, int):
        """
        Split image tensor into non-overlapping square tiles. 
        Tiles are ordered left-to-right and top-to-bottom:

        .. highlight:: text
        .. code-block:: text

            | 0 | 1 | 2 |
            -------------
            | 3 | 4 | 5 |
            -------------
            | 6 | 7 | 8 |

        .. warning:: 

            If image dimensions are not evenly divisible by tile size, 
            image will be effectively cropped, based on how many tiles can fit.

        :param im: Image tensor sized [N=1, C, H, W] or [C, H, W].
        :param shape: Tile shape (height, width) in pixels.
        :return: Image tensor sized [N, C, H, W], number of rows, number of columns.
        """
        ndim = len(im.size())
        assert ndim == 3 or (ndim == 4 and im.size(0) == 1), "tensor must be sized [N=1, C, H, W] or [C, H, W]"
        if ndim == 3: im = im.unsqueeze(0)
        C, H, W = im.shape[1:]

        rows = H // shape[0] #tile_size
        cols = W // shape[1] #tile_size

        im_tiles = im.unfold(2, shape[0], shape[0]).unfold(3, shape[0], shape[1])
        im_tiles = im_tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, shape[0], shape[1])
        tiles = torch.cat([tile.unsqueeze(0) for tile in im_tiles], dim=0)

        return tiles, rows, cols

    @classmethod
    def merge(cls, tiles:torch.Tensor, rows:int, cols:int) -> torch.Tensor:
        """
        Combine non-overlapping tiles into composite image tensor.
        Tiles are expected to be ordered left-to-right and top-to-bottom:

        .. highlight:: text
        .. code-block:: text

            | 0 | 1 | 2 |
            -------------
            | 3 | 4 | 5 |
            -------------
            | 6 | 7 | 8 |

        :param tiles: Tiles as pytorch image tensor sized [N, C, H, W].
        :param rows: Total number of rows in tile grid.
        :param cols: Total number of columns in tile grid.
        :return: Combined image tensor sized [C, H, W].
        """
        assert len(tiles.size()) == 4, "image tensor must be sized [N, C, H, W]"
        N, C, H, W = tiles.size()      
        assert rows * cols == N, "number of tiles must match rows * columns"
        combined_im = torch.zeros((1, C, rows * H, cols * W))

        for i in range(N):
            row = i // cols
            col = i % cols
            combined_im[..., row * H:(row + 1) * H, col * W:(col + 1) * W] = tiles[i:i+1,...]
        return combined_im.squeeze(0)

    @classmethod
    def blend(cls, 
        tiles:torch.Tensor, 
        rows:int=1, 
        cols:int=1, 
        wrap=True, 
        vector_data=False) -> torch.Tensor:
        """
        Blend tiles to remove seams.

        :param tiles: Tiles as pytorch image tensor sized [N, C, H, W] or [C, H, W].
        :param rows: Total number of rows in tile grid.
        :param cols: Total number of columns in tile grid.
        :param wrap: Wrap tile grid border (allows self-tiling).
        :param vector_data: Tiles contain directional unit vectors in [-1, 1] range - i.e. a normal map. 
            If True, vectors will be converted to angles for blending so that component gradients 
            can be matched independently.
        :return: Blended tiles sized [N, C, H, W] or [C, H, W].
        """
        ndim = len(tiles.size())
        assert ndim == 3 or ndim == 4, "tensor must be sized [C, H, W] or [N, C, H, W]"
        nobatch = ndim == 3
        if nobatch: tiles = tiles.unsqueeze(0)
        N, C, H, W = tiles.size()
        assert rows * cols == N, "number of tiles must match rows * columns"
        blended_tiles = torch.zeros_like(tiles)
        tiles_tmp = []
        if vector_data:
            assert C == 3, "vector data must have 3 channel components"
            for n in range(N):
                tiles_tmp.append(Geometry.normals_to_angles(tiles[n:n+1,...]))
            tiles_tmp = torch.cat(tiles_tmp, dim=0)
        else:
            tiles_tmp = tiles.clone()
        if N == 1:
            # make single image self-tiling
            for c in range(tiles_tmp.size(1)):
                blended_tiles[0, c:c+1] = cls.__poisson_blend(tiles_tmp[0, c:c+1], torch_tensors=True, cb=print)
        else:
            for n in range(N):
                pos_r, pos_c = cls.get_tile_position(n, rows, cols)
                t, r, b, l = cls.get_tile_neighbors(pos_r, pos_c, rows, cols, wrap=wrap)
                for c in range(tiles_tmp.size(1)):
                    blended_tiles[n, c:c+1] = cls.__poisson_blend(tiles_tmp[n, c:c+1],
                        torch_tensors=True,
                        top=tiles_tmp[t, c:c+1],
                        right=tiles_tmp[r, c:c+1],
                        bottom=tiles_tmp[b, c:c+1],
                        left=tiles_tmp[l, c:c+1],
                        cb=print)                    
        if vector_data:
            for n in range(N):
                blended_tiles[n:n+1] = Geometry.angles_to_normals(blended_tiles[n:n+1,...])
        return blended_tiles.squeeze(0) if nobatch else blended_tiles

    @classmethod        
    def __poisson_blend(cls,
        im:torch.Tensor, 
        top="self", 
        right="self", 
        bottom="self", 
        left="self", 
        torch_tensors=False, 
        solver=lss_enabled, 
        tol=1e-8,
        cb=print):
        im = im.permute(1, 2, 0).squeeze(2).numpy() if torch_tensors else im
        """
        Poisson solver for seamless textures - accepts one channel image
        input/output: tensor [C=1, H, W] / tensor [N=1, C=1, H, W] or np array [H, W, C] 
        """
        if torch_tensors:
            if top is not None and top != "self":       top = top.permute(1, 2, 0).squeeze(2).numpy()
            if right is not None and right != "self":   right = right.permute(1, 2, 0).squeeze(2).numpy()
            if bottom is not None and bottom != "self": bottom = bottom.permute(1, 2, 0).squeeze(2).numpy()
            if left is not None and left != "self":     left = left.permute(1, 2, 0).squeeze(2).numpy()
        
        # Masks
        H, W = im.shape[:2]
        m_white, m_boundary = np.ones((H, W)), np.ones((H, W))
        m_boundary[1:H-1,1:W-1] = 0
        m_inner = m_white - m_boundary

        # Pixel IDs
        id_px = np.arange(H * W).reshape(H, W)
        id_inner = id_px[np.nonzero(m_inner)].flatten()
        id_boundary = id_px[np.nonzero(m_boundary)].flatten()
        id_mask = id_px.flatten()

        # Matrix
        A = lil_matrix((len(id_mask), len(id_mask)))
        A[id_inner, id_inner - 1], A[id_inner, id_inner + 1], A[id_inner, id_inner - W], A[id_inner, id_inner + W] = 1, 1, 1, 1    
        A[id_boundary, id_boundary] = 1
        A[id_inner, id_inner] = -4
        A = A.tocsr()

        # Pixels to conform to
        bound = np.zeros_like(im)
        bound[0] =      ((im[0] + im[-1]) * 0.5             if isinstance(top, str) and top == "self" \
            else        (top[-1] + im[0]) * 0.5)            if top is not None else im[0]
        bound[-1] =     ((im[0] + im[-1]) * 0.5             if isinstance(bottom, str) and bottom == "self" \
            else        (bottom[0] + im[-1]) * 0.5)         if bottom is not None else im[-1]
        bound[:,0] =    ((im[:, 0] + im[:, -1]) * 0.5       if isinstance(left, str) and left =="self" \
            else        (left[:, -1] + im[:, 0]) * 0.5)     if left is not None else im[:, 0]
        bound[:,-1] =   ((im[:, 0] + im[:, -1]) * 0.5       if isinstance(right, str) and right =="self" \
            else        (right[:, 0] + im[:, -1]) * 0.5)    if right is not None else im[:, -1]

        # Laplacian and gradients
        lap_kern = np.array([
                [0,  1,  0],
                [1, -4,  1],
                [0,  1,  0]
            ])

        grad = scipy.signal.fftconvolve(im, lap_kern, mode="same")
        b = np.zeros(len(id_mask))
        b[id_inner] = grad[np.nonzero(m_inner)].flatten()
        b[id_boundary] = bound[np.nonzero(m_boundary)].flatten()

        # Poisson solver
        cb(f" / solving Poisson system with {lss_enabled}")
        ml, x = None, None
        if solver == "spsolve":
            x = lss(A, b).reshape(im.shape)
        elif solver == "pyamg":
            ml = lss.ruge_stuben_solver(A)
            x = ml.solve(b, tol=tol).reshape(im.shape) # tol 1e-10
        elif solver == "amgcl":
            P = lss.amg(A, prm={'relax.type': 'spai0'}) # spai0, ilu0
            solve = lss.solver(P, prm=dict(type='lgmres', tol=tol, maxiter=1000)) # ? bicgstab, lgmres ; tol 1e-8
            x = solve(A, b).reshape(im.shape)
        else:
            raise ImportError('failed to identify poisson solver')

        res = torch.from_numpy(x).unsqueeze(2).permute(2, 0, 1).unsqueeze(0) if torch_tensors else x
        return res