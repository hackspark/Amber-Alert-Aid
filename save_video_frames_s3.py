import cv2 # OpenCV 3.0.0
import boto
from aws_keys import get_aws_keys


class SaveVideoFramesS3(object):

    def __init__(self, video_path, get_aws_keys=get_aws_keys, bucket_name='amberalertsparkhackathon'):
        """
        Play video, do background subtraction on the images and save frames and
        background mask to S3

        :param video_path: The path to the traffic video (will do streaming in the future)
        :param get_aws_keys: Function that gives (aws_access_key, aws_secret_access_key)
        :param bucket_name: The bucket name to write to
        """
        self.video_path = video_path
        # Get the video name without the extension
        self.video_name = str(video_path.split('/')[-1].split('.')[0])

        # Set up S3 bucket
        self.aws_key, self.aws_secret_key = get_aws_keys()
        self.s3_conn = boto.connect_s3(self.aws_key, self.aws_secret_key)
        self.bucket_conn = self.create_bucket_if_not_exist(bucket_name)

    def create_bucket_if_not_exist(self, bucket_name):
        """
        Create an s3 bucket if it does not already exist, otherwise connect to it.
        Either way return the bucket connection

        :param bucket_name: The s3 bucket name
        :return: s3 bucket connection
        """
        if self.s3_conn.lookup(bucket_name) is None:
            b = self.s3_conn.create_bucket(bucket_name, policy='public-read')
        else:
            b = self.s3_conn.get_bucket(bucket_name)
        return b

    @staticmethod
    def get_bg_mask(frame, bg_subtractor):
        """
        Produce mask using background subtractor

        :param frame: The image frame as numpy array
        :param bg_subtractor: OpenCV background subtractor class instance
        :return: mask as numpy array
        """
        mask = bg_subtractor.apply(frame)
        return mask

    @staticmethod
    def get_bounding_box(frame, mask, bounding_shape):
        """
        This crops the frame and mask according to a bounding box that we have specified

        :param frame: The image frame that we are interested in
        :param mask: The background subtraction mask
        :param bounding_shape: The rectangular box in the image that we are interested in
        :return: bounding-box-frame, bounding-box-mask
        """

        x_min, x_max, y_min, y_max = bounding_shape
        bound_frame = frame[y_min:y_max, x_min:x_max]
        bound_mask = mask[y_min:y_max, x_min:x_max]
        return bound_frame, bound_mask

    def discard_image_decision_rule(self, frame, mask, bounding_shape=(1000, 1200, 650, 800)):
        """
        Decision rule to decide if bounding-box-frame and bounding-box-mask should be kept.
        If car in bounding box, keep, else toss

        :param frame: bounding-box-frame (numpy.array)
        :param mask: bounding-box-mask (numpy.array)
        :return: Boolean
        """
        bound_box_frame, bound_box_mask = self.get_bounding_box(frame, mask, bounding_shape)
        if bound_box_mask.mean() > .3 * 255.0:  # max value of mask pixel is 255
            return True
        else:
            return False

    def write_frame_mask_s3(self, frame, mask, ith_frame):
        """
        Create unique names and write frames and masks to S3 for import to Spark later

        :param frame: Image frame numpy array
        :param mask: Background mask numpy array
        """
        file_name = '%s_%s.txt' % (ith_frame, self.video_name)

        # Make the file objects for frame and mask
        frame_file = self.bucket_conn.new_key('frame/%s' % file_name)
        mask_file = self.bucket_conn.new_key('mask/%s' % file_name)

        # Write to the file objects for frame and mask as a list of list (as a string)
        frame_file.set_contents_from_string(str(frame.tolist()))
        mask_file.set_contents_from_string(str(mask.tolist()))

        print 'Written Frame and Mask', file_name

    def filter_video_save_s3(self):
        """
        Play the video, decides to keep the frame or not based on bounding box and finally
        save the frames we are going to keep to a designated s3 bucket
        """
        video = cv2.VideoCapture(self.video_path)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        ith_frame = 0
        while video.isOpened():
            success, frame = video.read()
            ith_frame += 1

            if success:
                mask = self.get_bg_mask(frame, bg_subtractor)
                self.write_frame_mask_s3(frame, mask, ith_frame)
            else:
                raise Exception('Cannot read the video / Done reading ...')

if __name__ == '__main__':
    save_vid = SaveVideoFramesS3('traffic_video1.mov', get_aws_keys=get_aws_keys,
                                 bucket_name='amberalertsparkhackathon')
    save_vid.filter_video_save_s3()
