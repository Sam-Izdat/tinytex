from multiResolution_textureSynthesis import *
from makeGif import *

parms = {
  "exampleMapPath": "../../../data/chain.png",
  "outputSize": [128, 128],
  "child_kernel_size": 5,
  "parent_kernel_size": 3,
  "saveImgsPath": "../../../out/33/",
  "saveGifPath": "../../../out/out33.gif",
  "pyramidLevels": 4,
  "pyramidType": "gaussian"
}

#user example will always be treated as level 0 (pass it to 'userExample' parameter)
userExample = {
    "userExamplePath": "userExample_4.jpg"
}

#run the texture synthesis
multiResolution_textureSynthesis(parms, userExample = None)

#Top row all generated levels
#Bottom row all example levels

#make a gif 
makeGif(parms["saveImgsPath"], parms["saveGifPath"], frame_every_X_steps = 2, repeat_ending = 10, start_frame = 0)