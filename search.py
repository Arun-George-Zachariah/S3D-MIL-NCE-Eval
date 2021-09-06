import os
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import math
import constants
import pickle


MODEL = None


def init():
    """Importing and Loading the model from TF-Hub."""
    global MODEL
    if MODEL is None:
        handle = constants.TF_HUB_URL
        MODEL = hub.load(handle)


def get_video_embeddings(model, input_frames):
  """Generate embeddings from the model for the video frames"""
  vision_output = model.signatures['video'](tf.constant(tf.cast(input_frames, dtype=tf.float32)))
  return vision_output['video_embedding']


def get_text_embeddings(model, input_words):
  """Generate embeddings from the model for the input words"""
  text_output = model.signatures['text'](tf.constant(input_words))
  return text_output['text_embedding']


def crop_center_square(frame):
    """Utility to open video files using CV2"""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]


def load_videos(video_dir, max_frames=32, resize=(224, 224)):
    """Loading videos from a directory"""
    video_lst = []
    file_lst = []
    for filename in os.listdir(video_dir):
        file_lst.append(filename)
        cap = cv2.VideoCapture(os.path.join(video_dir, filename))
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        frames = np.array(frames)
        if len(frames) < max_frames:
            n_repeat = int(math.ceil(max_frames / float(len(frames))))
            frames = frames.repeat(n_repeat, axis=0)
        frames = frames[:max_frames]
        video_lst.append(frames / 255.0)
    return file_lst, video_lst


def index(video_dir):
    """Indexing (i.e generating the embedding) a directory of videos."""
    init()
    file_lst, video_lst = load_videos(video_dir)
    videos_np = np.stack(video_lst, axis=0)
    video_embd = get_video_embeddings(MODEL, videos_np)
    os.makedirs(constants.DATA_DIR, exist_ok=True)
    with open(constants.DATA_DIR + "/index.pickle", "wb") as f:
        pickle.dump(video_embd, f)
    with open(constants.DATA_DIR + "/files.pickle", "wb") as f:
        pickle.dump(file_lst, f)


def retrieve(query_lst, video_embd, file_lst, top_k=1):
    """Obtaining the closest videos"""
    ret_lst = []
    words_np = np.array(query_lst)
    text_embd = get_text_embeddings(MODEL, words_np)
    all_scores = np.dot(text_embd, tf.transpose(video_embd))
    if len(all_scores) > 0 and top_k > len(all_scores[0]):
        print("Top k value greater that the number of files indexed")
        return
    for i in range(len(all_scores)):
        sorted_ix = np.argsort(-all_scores[i, :])
        res_lst = []
        for j in range(top_k):
            res_lst.append(file_lst[sorted_ix[j]])
        ret_lst.append(res_lst)
    return ret_lst


if __name__ == "__main__":
    query = ["waterfall", "playing guitar", "car drifting"]
    index("/mydata/S3D-MIL-NCE-Eval/Video_Data")
    with open(constants.DATA_DIR + "/index.pickle", "rb") as f:
        video_embd = pickle.load(f)
    with open(constants.DATA_DIR + "/files.pickle", "rb") as f:
        file_lst = pickle.load(f)
    results = retrieve(query, video_embd, file_lst, 3)
    print("Results: ", results)



