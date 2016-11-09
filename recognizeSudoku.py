import os,sys
import numpy as np
import cv2
import random
from PIL import Image
import pytesseract

#detect and cut the biggest box in the image
def get_sudoku_box(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)

	contours,hierarchy=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	

	boxes = []
	biggest = bx = by = bw = bh = 0

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if(w*h) > biggest:
			bc = cnt
			biggest = w*h
 			bx = x
 			by = y
 			bw = w
 			bh = h

 	#cv2.rectangle(gray,(bx,by),(bx+bw,by+bh),(0,255,0),2)
 	sudokubox = image[by:by+bh,bx:bx+bw]
 	return sudokubox

#get houghlines in the sudokubox
def get_sudoku_lines(sudokubox):
 	canny  = cv2.Canny(sudokubox,50,150,apertureSize = 3)

 	lin = []

 	lines = cv2.HoughLines(canny,1,np.pi/180,200)
	for rho,theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		lin.append([x1, y1, x2, y2])

	return lin

#divide into horizontal and vertical lines and transforme lines to inside the image
def split_lines(lines, sudokubox):
	horizontal_lines = []
	vertical_lines = []

	for l in lines:
		x1, y1, x2, y2 = l
		dx = abs(x1-x2)
		dy = abs(y1-y2)
		if dx > dy:
			dx = (x1-x2)
			dy = (y1-y2)
			if dx == 0:
				rate = 0
			else:	
				rate = float(dy)/float(dx)
			y1 = int(y1 - x1*rate)
			x1 = 0
			#which is the right size?
			y2 = int(y1 + sudokubox.shape[1]*rate)
			x2 = sudokubox.shape[1]
			horizontal_lines.append([x1,y1,x2,y2])
			
		else:
			dx = (x1-x2)
			dy = (y1-y2)
			if dx == 0:
				rate = 0
			else:	
				rate = float(dx)/float(dy)
			x1 = int(x1 - y1*rate)
			y1 = 0
			#which is te right size?
			x2 = int(x1 + sudokubox.shape[0]*rate)
			y2 = sudokubox.shape[0]
			vertical_lines.append([x1,y1,x2,y2])

	return [horizontal_lines, vertical_lines]

#put lines into groups that belong to the same line	
def group_lines(horizontal_lines, vertical_lines, sudokubox):

	horizontal_lines.sort(key=lambda x: x[1])
	prev = horizontal_lines[0][1]
	average = 0

	for hl in horizontal_lines:
		x1, y1, x2, y2 = hl
		average = average + abs(prev-y1)
		prev = y1
	average = average/len(horizontal_lines)

	hlinegroups = []
	first = 1
	group = []
	r = g = b = 0
	prev = horizontal_lines[0][1]
	for hl in horizontal_lines:
		x1, y1, x2, y2 = hl
		if abs(prev-y1) < average or first:
			first = 0
			group.append(hl)
		else:
			hlinegroups.append(group)
			group = []
			group.append(hl)
		prev = y1
	hlinegroups.append(group)

	vertical_lines.sort(key=lambda x: x[0])
	prev = vertical_lines[0][0]
	average = 0

	for vl in vertical_lines:
		x1, y1, x2, y2 = vl
		average = average + abs(prev-x1)
		prev = x1
	average = average/len(vertical_lines)

	vlinegroups = []
	first = 1
	prev = vertical_lines[0][0]
	group = []
	for vl in vertical_lines:
		x1, y1, x2, y2 = vl
		if abs(prev-x1) < average or first:
			first = 0
			group.append(vl)
		else:
			vlinegroups.append(group)
			group = []

			group.append(vl)
		prev = x1
	vlinegroups.append(group)

	return [hlinegroups ,vlinegroups]

#take the average line of a group of lines
def average_line_groups(groups):
	lines = []
	#average lines of linegroups
	for group in groups:
		totals = [sum(x) for x in zip(*group)]
		size = float(len(group))
		line = [int(total / size) for total in totals]
		lines.append(line)

	return lines 

#function to draw lines on the image with a random color
def draw_lines(lines, image):
	color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	#draw lines
	for line in lines:
		x1, y1, x2, y2 = line
		cv2.line(image,(int(x1),int(y1)),(int(x2),int(y2)), color,2)

	return image

#function to draw the number boxes on the image 
def draw_boxes(boxes, image):
	color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	#draw boxes
	for box in boxes:
		p1, p2 = box
		cv2.rectangle(image,(p1[0],p1[1]),(p2[0],p2[1]),color,2)
	return image

#gives intersection point of 2 lines
def intersect_lines(hline,vline):
	hline = [[hline[0],hline[1]],[hline[2],hline[3]]]
	vline = [[vline[0],vline[1]],[vline[2],vline[3]]]
	xdiff = (hline[0][0] - hline[1][0], vline[0][0] - vline[1][0])
	ydiff = (hline[0][1] - hline[1][1], vline[0][1] - vline[1][1])
	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		raise Exception('lines do not intersect')

	d = (det(*hline), det(*vline))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return [x, y]

#gives box coordinates by horizontal and vertical lines
def get_boxes(horizontal_lines, vertical_lines):
	boxes = []
	for x in xrange(9):
		for y in xrange(9):
			p1 = intersect_lines(horizontal_lines[x], vertical_lines[y])
			p2 = intersect_lines(horizontal_lines[x+1], vertical_lines[y+1])
			boxes.append([p1,p2])
	return boxes

#cut the boxes from the image
def cut_boxes(boxes, sudokubox):
	box_images = []
	for box in boxes:
		p1x, p1y = box[0] 
		p2x, p2y = box[1]
		box_image = sudokubox[p1y:p2y,p1x:p2x]
		box_images.append(box_image)
	return box_images

#get the number image from the box_image or return none if there is no number
def subtract_number(box_image):
	box_image = cv2.cvtColor(box_image,cv2.COLOR_BGR2GRAY)
	ret, box_image = cv2.threshold(box_image, 90, 255, cv2.THRESH_BINARY)

	w,h = list(box_image.shape)

	#cut 10% from the border
	offset_x = w/10
	offset_y = h/10
	number_image = box_image[offset_x:w-offset_x,offset_y:h-offset_y]
	w,h = list(number_image.shape)
	if (number_image == 0).sum() < 350:
		return
	side = min(w,h)
	number_image = number_image[0:side,0:side]
	number_image = number_image[10:80,10:80]
 	return number_image

def make_mock_sudoku(number_images):
	mocksudoku = np.zeros(81)
	numbers_image = np.full((70,0), 255.0)

	for n, number_image in enumerate(number_images):
		number_image = subtract_number(number_image)
		if number_image is not None:
			number_image = np.array(number_image)
			#set a 1 in the mocksudoku and join number_image to be ocr'ed image
			mocksudoku[n] = 1
			#cv2.imwrite(unicode(n) + 'image_name.jpeg', number_image)
			
			numbers_image = np.concatenate([numbers_image, number_image], axis=1)

	return mocksudoku, numbers_image
    	
def main():
	#read imagenames from the command line
	for image_name in sys.argv[1:]:
		print image_name
		image  = cv2.imread(image_name)
		sudokubox = get_sudoku_box(image)
		lines = get_sudoku_lines(sudokubox)
		h, v = split_lines(lines, sudokubox)
		hg, vg = group_lines(h, v, sudokubox)
		hl = average_line_groups(hg)
		vl = average_line_groups(vg)
		boxes = get_boxes(hl,vl)

		box_images = cut_boxes(boxes, sudokubox)

		mocksudoku, ocr_image = make_mock_sudoku(box_images)
		cv2.imwrite('output_ocr' + image_name, ocr_image)

		sudoku_digits = pytesseract.image_to_string(Image.fromarray(ocr_image.astype(np.uint8)))

		sudoku_digits = list(unicode(sudoku_digits))
		sudoku_digits.reverse()

		sudoku = []
		for x in mocksudoku:
			if x:
				sudoku.append(int(sudoku_digits.pop()))
			else:
				sudoku.append(0)


		for i in xrange(9):
			for j in xrange(9):
				sys.stdout.write(unicode(sudoku[i*9+j]))
			sys.stdout.write('\n')

		sudokubox = draw_boxes(boxes, sudokubox)
		cv2.imwrite('output' + image_name, sudokubox)

main()

