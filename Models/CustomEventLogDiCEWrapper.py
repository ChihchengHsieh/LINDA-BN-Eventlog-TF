import tensorflow as tf

class CustomEventLogDiCEWrapper(tf.keras.Model):
    '''
    The event log model for our own DiCE algorithm.
    '''

    def __init__(self, ):
        super(CustomEventLogDiCEWrapper, self).__init__()
        