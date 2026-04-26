from PIL import Image, ImageDraw, ImageFont

# Create a test image
img = Image.new('RGB', (300, 200), color='blue')
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 50)
except:
    font = ImageFont.load_default()
draw.text((50, 100), "Test", fill="white", font=font)
img.save("test.jpg")