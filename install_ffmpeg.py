#!/usr/bin/env python3
"""
FFmpeg Installation Helper for TikTok-Reddit MCP

This script helps install FFmpeg on Windows and Linux systems
for the TikTok to Reddit video processing pipeline.
"""

import platform
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_ffmpeg():
    """Check if FFmpeg is already installed and working."""
    print("🔍 Checking for existing FFmpeg installation...")

    success, stdout, stderr = run_command(["ffmpeg", "-version"])
    if success and "ffmpeg version" in stderr:
        print("✅ FFmpeg is already installed and working!")
        return True

    success, stdout, stderr = run_command(["ffprobe", "-version"])
    if success and "ffprobe version" in stderr:
        print("✅ FFprobe is already installed and working!")
        return True

    print("❌ FFmpeg not found or not working properly.")
    return False

def install_windows():
    """Install FFmpeg on Windows."""
    print("\n🪟 Windows FFmpeg Installation")
    print("=" * 40)

    # Check for package managers
    print("\n📦 Checking for package managers...")

    # Check Chocolatey
    choco_success, _, _ = run_command(["choco", "--version"])
    if choco_success:
        print("✅ Chocolatey detected!")
        print("Installing FFmpeg with Chocolatey...")
        print("Run: choco install ffmpeg")
        return

    # Check Scoop
    scoop_success, _, _ = run_command(["scoop", "--version"])
    if scoop_success:
        print("✅ Scoop detected!")
        print("Installing FFmpeg with Scoop...")
        print("Run: scoop install ffmpeg")
        return

    # Manual installation
    print("\n🔧 Manual Installation Instructions:")
    print("1. Download FFmpeg from: https://ffmpeg.org/download.html")
    print("2. Choose 'Windows builds' from gyan.dev")
    print("3. Download the latest release (ffmpeg-release-essentials.zip)")
    print("4. Extract to C:\\ffmpeg\\ or E:\\ffmpeg\\")
    print("5. Add the bin folder to your PATH environment variable")
    print("\n📁 Recommended extraction path: C:\\ffmpeg\\")
    print("📁 Your bin folder should be: C:\\ffmpeg\\bin\\")

def install_linux():
    """Install FFmpeg on Linux."""
    print("\n🐧 Linux FFmpeg Installation")
    print("=" * 40)

    # Detect distribution
    distro = ""
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r") as f:
            for line in f:
                if line.startswith("ID="):
                    distro = line.split("=")[1].strip().strip('"')
                    break

    print(f"📍 Detected distribution: {distro}")

    if distro in ["ubuntu", "debian", "linuxmint", "pop"]:
        print("\n🔧 Ubuntu/Debian Installation:")
        print("sudo apt update")
        print("sudo apt install ffmpeg")
        print("\nOr for latest version:")
        print("sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7")
        print("sudo apt update")
        print("sudo apt install ffmpeg")

    elif distro in ["centos", "rhel", "fedora"]:
        print("\n🔧 CentOS/RHEL/Fedora Installation:")
        if distro == "fedora":
            print("sudo dnf install ffmpeg")
        else:
            print("sudo yum install ffmpeg")
            print("\nOr for latest version:")
            print("sudo yum install epel-release")
            print("sudo rpm -v --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro")
            print("sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm")
            print("sudo yum install ffmpeg")

    elif distro == "arch":
        print("\n🔧 Arch Linux Installation:")
        print("sudo pacman -S ffmpeg")

    else:
        print("\n🔧 Generic Linux Installation:")
        print("1. Download from: https://ffmpeg.org/download.html")
        print("2. Choose your distribution")
        print("3. Follow the installation instructions")
        print("\nOr try Snap:")
        print("sudo snap install ffmpeg")

def install_macos():
    """Install FFmpeg on macOS."""
    print("\n🍎 macOS FFmpeg Installation")
    print("=" * 40)

    # Check for Homebrew
    brew_success, _, _ = run_command(["brew", "--version"])
    if brew_success:
        print("✅ Homebrew detected!")
        print("Installing FFmpeg with Homebrew...")
        print("brew install ffmpeg")
        return

    print("\n🔧 Manual Installation:")
    print("1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
    print("2. Then run: brew install ffmpeg")

def main():
    """Main installation function."""
    print("🎬 FFmpeg Installation Helper for TikTok-Reddit MCP")
    print("=" * 55)

    system = platform.system().lower()

    # Check if already installed
    if check_ffmpeg():
        print("\n🎉 FFmpeg is ready to use!")
        print("Your TikTok-Reddit MCP should work perfectly now.")
        return

    # Install based on platform
    if system == "windows":
        install_windows()
    elif system == "linux":
        install_linux()
    elif system == "darwin":
        install_macos()
    else:
        print(f"❌ Unsupported platform: {system}")
        print("Please visit https://ffmpeg.org/download.html for manual installation.")

    print("\n" + "=" * 55)
    print("📋 After installation:")
    print("1. Restart your terminal/command prompt")
    print("2. Run: python -c \"from video_service import VideoService; print('FFmpeg paths:', VideoService()._get_ffmpeg_paths())\"")
    print("3. Verify the paths are detected correctly")
    print("\n🎯 Your TikTok-Reddit MCP will automatically use FFmpeg once installed!")

if __name__ == "__main__":
    main()