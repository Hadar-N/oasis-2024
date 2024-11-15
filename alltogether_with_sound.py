import time
import cv2
import pygame
import pygame.midi
import numpy as np
import utils.consts as consts
from utils.dataSingleton import DataSingleton
from utils.setup_helpers import screenSetup, getTransformationMatrix, originImageResize
from utils.internals_management_helpers import AddSpritesToGroup, checkCollision, getFishOptions

cam = cv2.VideoCapture(0)
ret,image = cam.read()

if not ret:
    print("Error: Failed to capture image")

win_name = "threshold image"
win_contour_name = "contour board res"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(win_contour_name, cv2.WINDOW_NORMAL)

pygame.init()
clock = pygame.time.Clock()

pygame.midi.init()
player = pygame.midi.Output(0)

global_data = DataSingleton()

img_resize = originImageResize(image)
window_size, window_flags = screenSetup(img_resize)
global_data.window_size = window_size
window = pygame.display.set_mode(global_data.window_size, window_flags)
player.set_instrument(consts.INSTRUMENT)

internals = pygame.sprite.Group()

#TODO: fix lighting when most is darkened? fish recognize themselves as non-mask!!!!

def findContours(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_image, consts.BLUR_SIZE, 0)
    kernel = np.ones((5, 5), np.uint8)
    dialated = cv2.dilate(blurred_img, kernel, iterations=1)
    thresh_image = cv2.adaptiveThreshold(dialated,consts.THRESHOLD_MAX,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,consts.ADAPTIVE_THRESH_AREA,consts.ADAPTIVE_THRESH_CONST)
    cleaned_mask = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy

def createMask(reference_blurred, current_image, kernel):
    current_blurred = cv2.GaussianBlur(current_image, consts.BLUR_SIZE, 0)
    difference = cv2.absdiff(current_blurred, reference_blurred)
    _, thresholded = cv2.threshold(difference, consts.THRESHOLD_VAL, consts.THRESHOLD_MAX, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    return closed

def playSingleNote(note : int, len : float = 0, velocity : int = 127, isDone : bool = False):
    player.note_on(note, velocity)
    if isDone or len:
        mus_len = consts.QUARTER_LEN * len
        time.sleep(mus_len)
        player.note_off(note, velocity)

def calcNoteToPlay(area: int):
    maxArea = global_data.window_size[0] * global_data.window_size[1]
    freeArea = area/maxArea

    note = (freeArea * (consts.MAX_DO - consts.MIN_DO)) + consts.MIN_DO
    return int(note)

contours, _ = findContours(image)
board_pts, matrix = getTransformationMatrix(contours, img_resize, global_data.window_size)

reference_image = cv2.warpPerspective(image, matrix, global_data.window_size ,flags=cv2.INTER_LINEAR)
reference_blur = cv2.GaussianBlur(reference_image, consts.BLUR_SIZE, 0)
kernel = np.ones((11, 11), np.uint8)  # Larger kernel for more aggressive closing

fish_options = getFishOptions()
global_data.fish_options = fish_options
internals = pygame.sprite.Group()

board_cont = np.array(board_pts, dtype=np.int32).reshape((-1, 1, 2))
cv2.drawContours(image,[board_cont], -1, (0,0,255), 5)
cv2.imshow(win_contour_name, image)

# Main loop
running = True

while running:
    window.fill((0,0,0))

    _,image = cam.read()
    image = cv2.warpPerspective(image, matrix, global_data.window_size ,flags=cv2.INTER_LINEAR)

    mask_img = createMask(reference_blur, image, kernel)
    mask_img_transformed = pygame.transform.flip(pygame.surfarray.make_surface(np.rot90(mask_img)), True, False)
    mask = pygame.mask.from_threshold(mask_img_transformed, (0,0,0), threshold=(1, 1, 1))
    area = (global_data.window_size[0] * global_data.window_size[1]) - mask.count()

    AddSpritesToGroup(internals, mask, area)
    just_flipped = checkCollision(internals, mask)

    if(len(just_flipped)): [playSingleNote(calcNoteToPlay(area) + f.interval) for f in just_flipped]

    internals.update()
    internals.draw(window)
    
    # Update the Pygame display
    pygame.display.update()
    clock.tick(30)
    
    # gray_mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    # mask_contours, _ = cv2.findContours(gray_mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in mask_contours:
    #     cv2.drawContours(gray_mask_img, [contour], -1, (120,120,120), 5)

    cv2.imshow(win_name, mask_img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

cv2.destroyAllWindows()
cam.release()
del player
pygame.midi.quit()
pygame.quit()