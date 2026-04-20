import cv2
import numpy as np
import os
import argparse

class TemplateExtractor:
    def __init__(self, template_width=15, target_y=70):
        self.template_width = template_width
        self.target_y = target_y
        self.image = None
        self.clone = None
        self.gray_image = None
        self.current_template_1d = None
        self.save_count = 0

    def load_image(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: Could not find image at {filepath}")
            return False
            
        self.image = cv2.imread(filepath)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.clone = self.image.copy()
        
        # Draw the mandatory y=70 guide line
        cv2.line(self.clone, (0, self.target_y), (self.clone.shape[1], self.target_y), (0, 255, 0), 1)
        return True

    def click_and_crop(self, event, x, y, flags, param):
        # Trigger only on left mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            # Calculate the x boundaries to center the template on the click
            x_start = max(0, x - self.template_width // 2)
            x_end = min(self.image.shape[1], x + self.template_width // 2 + 1)
            
            # Extract the raw 1D mathematical signal from the GRAYSCALE image at y=70
            self.current_template_1d = self.gray_image[self.target_y:self.target_y+1, x_start:x_end]
            
            # --- Visual Feedback ---
            display_img = self.clone.copy()
            
            # Draw a red bounding box on the main image to show the selected area
            # (We make it 10 pixels high just so you can actually see it on screen)
            cv2.rectangle(display_img, (x_start, self.target_y - 5), (x_end, self.target_y + 5), (0, 0, 255), 1)
            
            # Extract a slightly taller visual patch for the preview window
            visual_patch = self.image[max(0, self.target_y-10):min(self.image.shape[0], self.target_y+10), x_start:x_end]
            
            # Scale up the tiny preview patch so it's readable on modern monitors
            preview_zoomed = cv2.resize(visual_patch, (300, 200), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow("Main Image", display_img)
            cv2.imshow("Template Preview (Zoomed)", preview_zoomed)
            
            print(f"Selected X: {x} | Width: {x_end - x_start}px. Press 's' to save or click again.")

    def run(self):
        cv2.namedWindow("Main Image")
        cv2.setMouseCallback("Main Image", self.click_and_crop)
        cv2.imshow("Main Image", self.clone)

        print("--- Instructions ---")
        print("1. Left-Click exactly on the center of the weld gap.")
        print("2. Look at the 'Template Preview' window to verify the crop.")
        print("3. Press 's' to save the template to disk.")
        print("4. Press 'q' to quit the tool.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'q' to quit
            if key == ord("q"):
                break
                
            # Press 's' to save the template
            elif key == ord("s"):
                if self.current_template_1d is not None:
                    os.makedirs("templates", exist_ok=True)
                    
                    filename = os.path.join("templates", f"template_{self.template_width}px_{self.save_count}.npy")
                    # Increment save_count to avoid overwriting existing templates
                    while os.path.exists(filename):
                        self.save_count += 1
                        filename = os.path.join("templates", f"template_{self.template_width}px_{self.save_count}.npy")

                    # Save as a raw numpy array, NOT a jpg
                    np.save(filename, self.current_template_1d)
                    print(f"[SUCCESS] Saved 1D template to {filename}")
                    self.save_count += 1
                else:
                    print("[WARNING] Click on the image first to select a template!")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can change the image name and desired template width here
    IMAGE_TO_LOAD = "WeldGapImages_export/Set 2/image0207.jpg" # Replace with a clean image from Set 1
    TEMPLATE_WIDTH = 15        # 6px gap + surrounding metal
    
    tool = TemplateExtractor(template_width=TEMPLATE_WIDTH,target_y=70)
    if tool.load_image(IMAGE_TO_LOAD):
        tool.run()