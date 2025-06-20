#!/bin/bash

# Paper Compilation Script for arXiv Submission
# This script compiles the LaTeX paper and prepares it for submission

echo "📄 Paper Compilation Script"
echo "=========================="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install a LaTeX distribution:"
    echo "   - macOS: brew install --cask mactex"
    echo "   - Ubuntu: sudo apt-get install texlive-full"
    echo "   - Windows: Install MiKTeX from https://miktex.org/"
    echo ""
    echo "Alternatively, use Overleaf (https://overleaf.com) for online compilation."
    exit 1
fi

# Create output directory
mkdir -p output

echo "🔧 Compiling LaTeX document..."

# First compilation
pdflatex -output-directory=output limitations_of_nerf_few_shot.tex

# Second compilation for references
pdflatex -output-directory=output limitations_of_nerf_few_shot.tex

# Check if compilation was successful
if [ -f "output/limitations_of_nerf_few_shot.pdf" ]; then
    echo "✅ Paper compiled successfully!"
    echo "📁 Output file: output/limitations_of_nerf_few_shot.pdf"
    
    # Get file size
    filesize=$(du -h output/limitations_of_nerf_few_shot.pdf | cut -f1)
    echo "📊 File size: $filesize"
    
    # Check for common issues
    echo ""
    echo "🔍 Checking for common issues..."
    
    # Check for undefined references
    if grep -q "LaTeX Warning: Reference.*undefined" output/limitations_of_nerf_few_shot.log; then
        echo "⚠️  Warning: Some references may be undefined"
    else
        echo "✅ All references appear to be defined"
    fi
    
    # Check for overfull boxes
    if grep -q "Overfull" output/limitations_of_nerf_few_shot.log; then
        echo "⚠️  Warning: Some text may overflow margins"
    else
        echo "✅ No text overflow detected"
    fi
    
    # Check for missing figures
    if grep -q "LaTeX Warning: File.*not found" output/limitations_of_nerf_few_shot.log; then
        echo "⚠️  Warning: Some figures may be missing"
    else
        echo "✅ All figures appear to be present"
    fi
    
else
    echo "❌ Compilation failed. Check the log file for errors."
    exit 1
fi

echo ""
echo "📋 arXiv Submission Checklist:"
echo "=============================="
echo "✅ Paper compiled successfully"
echo "📄 PDF file created: output/limitations_of_nerf_few_shot.pdf"
echo ""
echo "📝 Next steps for arXiv submission:"
echo "1. Review the PDF file"
echo "2. Add your name and affiliation to the paper"
echo "3. Add any missing figures"
echo "4. Create arXiv account at https://arxiv.org"
echo "5. Submit the .tex file and figures to arXiv"
echo ""
echo "📚 Categories for arXiv:"
echo "   - Primary: cs.CV (Computer Vision and Pattern Recognition)"
echo "   - Secondary: cs.LG (Machine Learning), cs.AI (Artificial Intelligence)"
echo ""
echo "🏷️  Suggested keywords:"
echo "   NeRF, Few-shot Learning, 3D Reconstruction, DINO, LoRA, Neural Radiance Fields"
echo ""
echo "🎯 Paper ready for submission! 🎯" 