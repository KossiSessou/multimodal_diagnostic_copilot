#!/bin/bash

# Define directories
DATA_DIR="data"
REPORTS_DIR="$DATA_DIR/reports"
IMAGES_DIR="$DATA_DIR/images"

# Create directories
mkdir -p "$REPORTS_DIR"
mkdir -p "$IMAGES_DIR"

echo "--- Starting Data Acquisition ---"

# 1. Download Clinical Notes (Reports)
echo "Downloading clinical notes (XML)..."
curl -L -o "$REPORTS_DIR/reports.tgz" "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"

# 2. Download Images (PNG)
echo "Downloading chest X-ray images (PNG)..."
curl -L -o "$IMAGES_DIR/images.tgz" "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"

echo "--- Extracting Data ---"

# Extract Reports
echo "Extracting reports..."
tar -xzf "$REPORTS_DIR/reports.tgz" -C "$REPORTS_DIR"
rm "$REPORTS_DIR/reports.tgz"

# Extract Images
echo "Extracting images..."
tar -xzf "$IMAGES_DIR/images.tgz" -C "$IMAGES_DIR"
rm "$IMAGES_DIR/images.tgz"

echo "--- Phase 2: Download Complete ---"
echo "Reports extracted to: $REPORTS_DIR"
echo "Images extracted to: $IMAGES_DIR"
