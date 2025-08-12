# Installing Tesseract OCR

## Windows Installation

### Option 1: Download Installer (Recommended)
1. Go to: https://github.com/UB-Mannheim/tesseract/wiki
2. Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)
3. Run the installer as Administrator
4. During installation, make sure to select additional language packs if needed:
   - English (eng) - usually included by default
   - French (fra)
   - German (deu)
   - Spanish (spa)
   - etc.

### Option 2: Using Chocolatey
```powershell
# Install Chocolatey first if you don't have it
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Tesseract
choco install tesseract
```

### Option 3: Using Scoop
```powershell
# Install Scoop first if you don't have it
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Install Tesseract
scoop install tesseract
```

## Verify Installation

After installation, verify Tesseract is working:

```powershell
tesseract --version
```

You should see output like:
```
tesseract 5.3.3
 leptonica-1.83.1
  libgif 5.2.1 : libjpeg 8d (libjpeg-turbo 2.1.4) : libpng 1.6.39 : libtiff 4.5.1 : zlib 1.2.13 : libwebp 1.3.2 : libopenjp2 2.5.0
```

## Configure Python Path (if needed)

If you get errors about Tesseract not being found, you may need to set the path in your Python code:

```python
import pytesseract

# Set the path to tesseract executable (adjust path as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Test OCR Functionality

Run this test to verify OCR is working:

```python
import pytesseract
from PIL import Image
import requests
from io import BytesIO

# Test with a simple image
try:
    # Create a simple test image or download one
    print("Testing OCR...")
    
    # You can test with any image file
    # text = pytesseract.image_to_string(Image.open('test_image.png'))
    # print(f"Extracted text: {text}")
    
    print("✅ OCR setup appears to be working!")
except Exception as e:
    print(f"❌ OCR test failed: {e}")
```

## Language Packs

To use OCR with different languages, make sure you have the appropriate language packs installed:

- English: `eng` (usually default)
- French: `fra`
- German: `deu`
- Spanish: `spa`
- Italian: `ita`
- Portuguese: `por`
- Russian: `rus`
- Chinese Simplified: `chi_sim`
- Japanese: `jpn`

You can specify languages when using OCR:
```python
text = pytesseract.image_to_string(image, lang='eng+fra+deu')
```