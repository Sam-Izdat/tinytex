import subprocess

def main_cli():
	print("Running tinycio setup...")
	subprocess.run(["tcio-setup"])

if __name__ == '__main__':
    main_cli()