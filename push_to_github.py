"""
GitHub Upload Helper Script
Assists in pushing this local repository to your remote GitHub account.
"""

import sys
import os
import subprocess

CWD = os.path.dirname(os.path.abspath(__file__))
GIT_EXE = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Git', 'cmd', 'git.exe')
if not os.path.exists(GIT_EXE):
    GIT_EXE = 'git'

def run_cmd(args):
    print(f"> {' '.join(args)}")
    res = subprocess.run(args, cwd=CWD)
    return res.returncode

def main():
    print("=" * 60)
    print(" GITHUB REPOSITORY UPLOAD HELPER")
    print("=" * 60)

    if len(sys.argv) > 1:
        remote_url = sys.argv[1].strip()
    else:
        print("\nEnter your GitHub repository URL (e.g., https://github.com/username/speech-enhancement.git):")
        remote_url = input("Repository URL: ").strip()

    if not remote_url:
        print("\nNo remote URL provided. If you want to create a new repo via GitHub CLI, run:")
        print("  gh auth login")
        print("  gh repo create speech-enhancement-prototype --public --source=. --push")
        return

    # Add remote
    subprocess.run([GIT_EXE, 'remote', 'remove', 'origin'], cwd=CWD, stderr=subprocess.DEVNULL)
    run_cmd([GIT_EXE, 'remote', 'add', 'origin', remote_url])
    run_cmd([GIT_EXE, 'branch', '-M', 'main'])

    print(f"\nPushing to {remote_url} ...")
    ret = run_cmd([GIT_EXE, 'push', '-u', 'origin', 'main'])

    if ret == 0:
        print("\n🎉 SUCCESS! All files have been uploaded to your GitHub repository.")
    else:
        print("\n⚠️ Push encountered an authentication or remote error.")
        print("Ensure you have created the repository on GitHub and have write permissions.")
        print("To authenticate using Personal Access Token:")
        print("  git push https://<YOUR_TOKEN>@github.com/<username>/<repo>.git main")

if __name__ == "__main__":
    main()
