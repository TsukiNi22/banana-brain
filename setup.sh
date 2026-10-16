#!/bin/bash
set -euo pipefail

BASE_URL="https://tsukini22.github.io/banana-brain"
KEY_NAME="RPM-GPG-KEY-tsukini"
PACKAGE="libbanana-brain"
SUDO="sudo"
INSTALL=true

# =========================
# Parse arguments
# =========================
usage() {
    echo "Usage: $0 [--no-sudo] [--pre] [--no-install]"
    echo "  --no-sudo      Run without sudo (requires the script to be run as root already)"
    echo "  --pre          Install the pre-release/unstable package (libbanana-brain-pre)"
    echo "  --no-install   Only setup the mirror & gpg key, don't install the package"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --no-sudo)
            SUDO=""
            ;;
        --pre)
            PACKAGE="libbanana-brain-pre"
            ;;
        --no-install)
            INSTALL=false
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            usage
            ;;
    esac
done

if [[ -z "$SUDO" ]] && [[ "$(id -u)" -ne 0 ]]; then
    echo "Error: --no-sudo requires the script to be run as root." >&2
    exit 1
fi

# =========================
# Requirements
# =========================
require() {
    for cmd in "$@"; do
        if ! command -v "$cmd" > /dev/null 2>&1; then
            echo "Error: '$cmd' is required but not installed." >&2
            exit 1
        fi
    done
}
require curl

# =========================
# Find the os
# =========================
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS_ID="${ID:-unknown}"
    OS_ID_LIKE="${ID_LIKE:-}"
    OS_VERSION="${VERSION_ID:-unknown}"
else
    echo "Error: /etc/os-release not found, cannot detect OS." >&2
    exit 1
fi

is_rpm_based() {
    [[ "$OS_ID" =~ ^(fedora|rhel|centos|rocky|almalinux)$ ]] || [[ "$OS_ID_LIKE" =~ (fedora|rhel) ]]
}

is_deb_based() {
    [[ "$OS_ID" =~ ^(debian|ubuntu)$ ]] || [[ "$OS_ID_LIKE" =~ (debian|ubuntu) ]]
}

# =========================
# RPM
# =========================
install_rpm_repo() {
    echo "Detected RPM-based system (ID=$OS_ID). Setting up banana-brain repo..."

    # The mirror is only built for a fixed release (see: .github/workflows/build-packages.yml)
    if [[ "$OS_ID" = "fedora" ]] && ! curl -fsSL -o /dev/null "$BASE_URL/fedora/$OS_VERSION/$(uname -m)/repodata/repomd.xml"; then
        echo "Warning: no packages published for fedora $OS_VERSION ($(uname -m)), the installation will probably fail." >&2
    fi

    $SUDO curl -fsSL -o /etc/yum.repos.d/banana-brain.repo \
        "$BASE_URL/banana-brain.repo"

    $SUDO rpm --import "$BASE_URL/$KEY_NAME"

    echo "banana-brain repo installed."
}

install_rpm_package() {
    local pm="dnf"
    command -v dnf > /dev/null 2>&1 || pm="yum"

    echo "Installing $PACKAGE..."
    $SUDO "$pm" install -y "$PACKAGE"
}

# =========================
# DEB
# =========================
install_deb_repo() {
    echo "Detected DEB-based system (ID=$OS_ID). Setting up banana-brain repo..."
    require gpg dpkg

    $SUDO install -m 0755 -d /etc/apt/keyrings

    curl -fsSL "$BASE_URL/$KEY_NAME" \
        | gpg --dearmor | $SUDO tee /etc/apt/keyrings/banana-brain.gpg > /dev/null
    $SUDO chmod a+r /etc/apt/keyrings/banana-brain.gpg

    ARCH="$(dpkg --print-architecture)"

    echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/banana-brain.gpg] ${BASE_URL}/debian stable main" \
        | $SUDO tee /etc/apt/sources.list.d/banana-brain.list > /dev/null

    $SUDO apt-get update

    echo "banana-brain repo installed."
}

install_deb_package() {
    echo "Installing $PACKAGE..."
    $SUDO apt-get install -y "$PACKAGE"
}

# =========================
# Dispatch
# =========================
if is_rpm_based; then
    install_rpm_repo
    if $INSTALL; then install_rpm_package; fi
elif is_deb_based; then
    install_deb_repo
    if $INSTALL; then install_deb_package; fi
else
    echo "Error: unsupported distribution (ID=$OS_ID, ID_LIKE=$OS_ID_LIKE)." >&2
    echo "Supported: Fedora/RHEL-based (rpm) and Debian/Ubuntu-based (deb)." >&2
    exit 1
fi

if $INSTALL; then
    echo "$PACKAGE installed. Next step: find_package(banana-brain) + banana-brain::banana-brain in your CMake"
elif is_rpm_based; then
    echo "You can now run: $SUDO dnf install $PACKAGE"
else
    echo "You can now run: $SUDO apt install $PACKAGE"
fi
