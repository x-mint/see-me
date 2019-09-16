from PIL import Image
from io import BytesIO

import sys
import base64

def manipulate_image(file_name):

	picture = Image.open("images/uploads/" + file_name)

	# get the size of the image
	width, height = picture.size

	# process every pixel
	for x in range(width):
		for y in range(height):
			current_color = list( picture.getpixel( (x,y) ) )
			# increase red, x2
			current_color[0] = min( int(current_color[0]) * 2, 255 )
			new_color = tuple(current_color)
			picture.putpixel( (x,y), new_color )

	# resize
	#new_width  = 300
	#new_height = 300
	#picture = picture.resize((new_width, new_height), Image.ANTIALIAS)

	#picture.save("images/model_outputs/" + file_name) ##

	buffered = BytesIO()
	picture.save(buffered, format="JPEG")
	picture_str = base64.b64encode(buffered.getvalue())
	print(picture_str)
	sys.stdout.flush()

if __name__ == "__main__":
	file_name = sys.argv[1]
	manipulate_image(file_name)
