# ScreenshotFramer

Frame iPhone screenshots and videos for instagram posts.

Download device frames from https://developer.apple.com/design/resources/#product-bezels

Only tested with iPhone 17pro portrait screenshots, but probably works with others.

## Usage

This is a command line tool. It takes 4 arguments:

1. Output Directory
2. Background image
3. Device frame
4. One or more screenshots or videos

The script automatically detects whether each input is an image or video file and processes it accordingly.

### Example usage for images:

```bash
./frame.py --out-dir ~/Desktop/post/rendered ~/Desktop/post/background.jpg frames/iphone-17/iPhone\ 17\ Pro/iPhone\ 17\ Pro\ -\ Cosmic\ Orange\ -\ Portrait.png ~/Desktop/post/*.png
```

### Example usage for videos:

```bash
./frame.py --out-dir ~/Desktop/post/rendered ~/Desktop/post/background.jpg frames/iphone-17/iPhone\ 17\ Pro/iPhone\ 17\ Pro\ -\ Cosmic\ Orange\ -\ Portrait.png ~/Desktop/post/demo.mp4
```

### Mix images and videos:

```bash
./frame.py --out-dir ~/Desktop/post/rendered ~/Desktop/post/background.jpg frames/iphone-17/iPhone\ 17\ Pro/iPhone\ 17\ Pro\ -\ Cosmic\ Orange\ -\ Portrait.png ~/Desktop/post/screen1.png ~/Desktop/post/demo.mp4
```

Supported video formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.webm`, `.flv`, `.wmv`

Video processing preserves the original audio and frame rate.

Example output:

https://www.instagram.com/p/DPWi-7gEUP6/?img_index=1
