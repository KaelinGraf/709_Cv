import cv2
import numpy as np
from sklearn import linear_model
import os
import matplotlib.pyplot as plt

CWD = os.getcwd()
PIX_WIDTH = 0.04607 #in mm, width of each pixel
MAX_WELD_WIDTH = 0.5 #in mm
MAX_DEV_H = 2.0 #maximum horizontal deviation speed in mm


DEBUG = False


class WeldDetector:
    def __init__(self,y_line:int=70, expected_width:int = 6, tolerance:int = 3,templates = [],images = []):
        """
        args:
            y_line (int): Line of expected weld center location in pixels
            expected_width (int): expected weld width in pixels
            tolerance (int): tolerance of prediction
            templates([str]): list of relative paths to template npy files
        """
        self._y_line:int = y_line
        self._expected_width:int= expected_width
        self._tolerance:int = tolerance
        self.templates = []
        self.images = []
        for image in images:
            self.images.append(self.load_image(image))
        self.load_templates(templates=templates)

        
    def load_image(self,path:str=None):
    
        if path is not None:
            img = cv2.imread(os.path.join(CWD,path))
            if img is None:
                raise Exception("File could not be loaded")

        return img #np array
    def crop_to_roi(self,img,height:int,width:int,x_center:int):
        """
        Crops image to ROI around weld center line
        args:
            img(np.array): image loaded by opencv
            height(int): height of crop (pixels). Should be divisible by two
            width(int): width of crop (pixels). Should be divisible by two
            x_center(int):approximate x location of the weld (pixels)
        """
        if height%2 != 0 or width%2!=0:
            raise Exception("ROI sizes not divisible by two")
        if height/2 > self._y_line:
            raise Exception(f"ROI height exceeds image bounds, must be less than {self._y_line*2}")
        w,h = img.size

        return None

    def rgb_to_greyscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    def crop_to_weld_line(self,img_gray:np.ndarray,y_line:int) -> np.ndarray:
        """
        crop image to weld line to extract approximate ROI
        args:
            img_gray(np.ndarray): greyscale image
            y_line(int): line of the array (from the top) to extract from

        returns:
            weld_line_slice(np.ndarray): Horizontal image slice taken at y_line 
        """
        return img_gray[y_line, :]
    
    def pix_to_world(self):
        pass
    def draw_weld(self, image: np.ndarray, predicted_center: int, y_line: int = 70) -> np.ndarray:
        """
        Draws a cross at the predicted weld center.
        args:
            image (np.ndarray): The image to draw on
            predicted_center (int): The predicted x coordinate of the weld center
        returns:
            image (np.ndarray): The image with the cross drawn
        """
        cv2.drawMarker(image, (int(predicted_center), y_line), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        return image
    

class WeldDetectorTemplateMatching(WeldDetector):
    def __init__(self, y_line = 70, expected_width = 6, tolerance = 3, templates=[], images=[]):
        super().__init__(y_line, expected_width, tolerance, templates, images)

    def extract_weld_center_from_slice(self,weld_line_slice:np.ndarray) -> int:
        """
        Extracts the pixel location of the weld center by treating weld_line_slide as a discreet signal, and performing zero-mean normalized cross-correlation (removes dc bias) with "template" weld signals
        This process is repeated for multiple templates.
        They then undergo the following process:
        1. recification (negative correlation is the inverse of the gradient we are looking for)
        2. square the signal to filter noise (supresses smaller peaks)
        3. Normalize by the sum of the array to represent a probability distribution
        4. Calculate the expected value mu_x = sum(x*P(x)) for each. 
        5. Take the median over mu_x of each template, reject any that lie +-2std out of the median (simple outlier removal)
        6. Use a Naive Bayesian Update (Product of Experts) with raised epsilon to combine distributions
        7. Take the argmax (peak probability) of the combined distribution
        """
        pdfs = []
        expected_values=[]
        if self.templates.size == 0:
            raise Exception("Templates not found, please run self.load_templates")
        for template in self.templates:
            #Perform cross-correlation
            res = cv2.matchTemplate(weld_line_slice,template,cv2.TM_CCOEFF_NORMED)
            res = res.flatten()
            #rectify and square signal to remove unwanted samples
            res_rectified = np.maximum(0,res)
            res_filtered = res_rectified ** 2
            # Normalize into probability distribution function
            area = np.sum(res_filtered)
            if area > 0:
                pdf = res_filtered / area
                
                x_indices = np.arange(len(pdf))

                #Gaussian blur on the PDF. This widens the receptive field of each peak such that Product of Experts does not fail as badly due to thin peaks
                sigma = 3.0
                blur_kernel_size = 15 
                pdf_2d = pdf.reshape(1, -1).astype(np.float32)
                pdf_blurred = cv2.GaussianBlur(pdf_2d, (blur_kernel_size, 1), sigmaX=sigma).flatten()
                
                # Re-normalize one final time to ensure it remains a valid probability distribution
                pdf = pdf_blurred / np.sum(pdf_blurred)
                
                # Recalculate mu after weighting/blurring for the outlier removal step
                final_mu = np.sum(x_indices * pdf)
                
                pdfs.append(pdf)
                expected_values.append(final_mu)
             
        #cast to fixed-sized np array for performance
        pdfs = np.array(pdfs)
        expected_values= np.array(expected_values)
        #take median of dataset (for outlier removal)
        ev_median = np.median(expected_values)
        ev_std = np.std(expected_values)
        if DEBUG:
            plt.figure(figsize=(12, 6))
            
            # Use a colormap to give each template a unique color
            colors = plt.cm.tab10(np.linspace(0, 1, len(pdfs)))
            
            for i, (pdf, mu) in enumerate(zip(pdfs, expected_values)):
                color = colors[i % len(colors)]
                
                # Plot the distribution curve
                plt.plot(pdf, label=f"Template {i} PDF", color=color, alpha=0.7)
                
                # Draw a vertical dashed line for the expected value (mu)
                plt.axvline(x=mu, color=color, linestyle='--', alpha=0.8, 
                            label=f"T{i} EV: {mu:.1f}")
                
                # Add a small dot right on the x-axis for visibility
                plt.plot(mu, 0, marker='^', markersize=8, color=color)

            plt.title("Weld Template Probability Distributions (Overlaid)")
            plt.xlabel("Pixel Coordinate (Shifted by Template Width)")
            plt.ylabel("Probability")
            # Place legend outside the plot so it doesn't cover the peaks
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        median_mask = (expected_values >= ev_median - (2*ev_std)) & (expected_values <= ev_median + (2*ev_std)) #remove and pdfs with an expected value over 2 standard deviations from the mean
        filtered_pdfs = pdfs[median_mask]
        # 6. Use Naive Bayesian Update (Product of Experts) 
        # A generous baseline epsilon allows templates to vote to suppress noise 
        # without acting as an absolute black-hole veto on a single mismatch.
        epsilon = 1e-15
        log_pdfs = np.log(filtered_pdfs + epsilon) #Conversion to log space allows for easy addition
        summed_logs = np.sum(log_pdfs, axis=0)
        unnormalized_combined = np.exp(summed_logs)
        final_pdf = unnormalized_combined / np.sum(unnormalized_combined) #convert to probability distribution
        if DEBUG:
            plt.figure(figsize=(12,6))
            plt.plot(final_pdf)
            plt.title("Final Probability Distribution")
            plt.xlabel("Pixel Coordinate (Shifted by Template Width)")
            plt.ylabel("Probability")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # 7. Take the argmax (peak probability) instead of Expected Value. Center-of-mass is vulnerable to being pulled off-center by random distant false-positive noise peaks.
        return int(np.argmax(final_pdf))

    def load_templates(self, template_dir="templates", templates=None, load_all=True):
        """
        Loads templates from either specified paths, or from directory search (or both if load_all is True)
        args:
            templates [str]: Local/relative path directories assuming CWD is the root
            load_all [bool]: Load all files (list and directory)
        """
        if templates is None:
            templates = []
            
        full_template_dir = os.path.join(CWD, template_dir)
        if not os.path.exists(full_template_dir):
            os.makedirs(full_template_dir, exist_ok=True)
            
        # Get all files inside template_dir and prefix them so np.load can find them
        dir_files = [os.path.join(template_dir, file) for file in os.listdir(full_template_dir) if "template" in file and ".npy" in file]

        if load_all:
            files_set = set(dir_files)
            if templates:
                files_set.update(templates)
            for file in files_set:
                self.templates.append(np.load(file)[0])
        else:
            if templates:
                files_list = set(templates)
            else:
                files_list = set(dir_files)
            for file in files_list:
                self.templates.append(np.load(file)[0])
        
        self.templates = np.asarray(self.templates)
        print(f"Loaded {len(self.templates)} templates.")

    def extract_weld_center_line_fitting(self,image_greyscale: np.ndarray,eval_lines: np.ndarray, linefit_algo = "RANSAC"):
        """
        Extracts the pixel location of the weld center via performing statistical (cross correlation based) prediction at multiple y-lines, then performing outlier-robust line fitting to regress weld location at y-line (70)
        args:
            image_greyscale (np.ndarray): [h x w x 1] image (call rgb_to_greyscale on input image)
            eval_lines (np.ndarray): lines to evaluate weld center for line fitting. More is better, and a wide range is ideal to fit a more accurate line 
            linefit_algo (str): Algorithm selection for line-fitting. NOTE: FILL IN DETAILS OF EACH ALGO HERE ONCE IMPLIMENTED 
        """
        algos = ["RANSAC", "LINEAR", "WLEASTSQUARES"]
        if linefit_algo not in algos: raise Exception(f"Linefit algo invalid, choose from {algos}")
        if any(eval_lines>image_greyscale.shape[1]): raise Exception(f"target eval line exceeds image dimensions of {image_greyscale.shape[0]} x {image_greyscale.shape[1]}")
        pred_centers = []
        for eval_line in eval_lines:
            weld_crop = self.crop_to_weld_line(image_greyscale,int(eval_line))
            pred_centers.append(self.extract_weld_center_from_slice(weld_line_slice=weld_crop)) #This is the result of per-template zero mean normalized cross correlation + bayesian update

        print(f"pred center = {pred_centers}")
        if linefit_algo == "RANSAC":
            ransac = linear_model.RANSACRegressor()
            ransac.fit(eval_lines.reshape(-1,1),pred_centers)
            prediction = ransac.predict(np.array([70]).reshape(-1,1))
            return prediction[0]
        elif linefit_algo == "LINEAR":
            linear = linear_model.LinearRegression()
            linear.fit(eval_lines.reshape(-1,1),pred_centers)
            prediction = linear.predict(np.array([70]).reshape(-1,1))
            return prediction[0]
        elif linefit_algo == "WLEASTSQUARES":
            wleast_squares = linear_model.Ridge()
            wleast_squares.fit(eval_lines.reshape(-1,1),pred_centers)
            prediction = wleast_squares.predict(np.array([70]).reshape(-1,1))
            return prediction[0]
        

        
   

    def process_image(self,image:np.ndarray):
        eval_lines=np.arange(0,140,5)
        pred_centers = self.extract_weld_center_line_fitting(self.rgb_to_greyscale(img=image),eval_lines=eval_lines,linefit_algo = "RANSAC")
        image  = self.draw_weld(image,int(pred_centers),70)

        cv2.imshow("Weld Center", image)
        cv2.waitKey(0)

    def process_all(self):
        for image in self.images:
            self.process_image(image)


def main():
    images = ['WeldGapImages_export/Set 1/image0012.jpg']
    weld_detector = WeldDetectorTemplateMatching(images = images)
    weld_detector.process_all()




if __name__ == "__main__":
    main()