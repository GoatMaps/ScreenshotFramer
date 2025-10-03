# ScreenshotFramer

Frame iPhone screenshots for instagram posts.

Download device frames from https://developer.apple.com/design/resources/#product-bezels

Only tested with iPhone 17pro portrait screenshots, but probably works with others.

## Usage

This is a command line tool. It takes 4 arguments:

1. Output Directory
2. Background image
3. Device frame
4. One or more screenshots

Example usage:

```bash
./frame.py --out-dir ~/Desktop/post/rendered ~/Desktop/post/background.jpg frames/iphone-17/iPhone\ 17\ Pro/iPhone\ 17\ Pro\ -\ Cosmic\ Orange\ -\ Portrait.png ~/Desktop/post/*.png
```

Example output:

https://www.instagram.com/p/DPWi-7gEUP6/?img_index=1
