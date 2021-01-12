import tensorflow as tf
import transformers
from transformers import TFBertForTokenClassification, TFXLMRobertaForTokenClassification


if __name__ == "__main__":

    model = TFBertForTokenClassification.from_pretrained("../norbert3/model.ckpt-1060000.data-00000-of-00001", from_tf=True)
