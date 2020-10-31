import tensorflow as tf

class StyleContentModel(tf.keras.models.Model):

    def clip_0_1(self,image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


    def gram_matrix(self,input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def vgg_layers(self,layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def __init__(self, style_layers, content_layers,content_weight):
        super(StyleContentModel, self).__init__()
        #Hyperparameters
        #self.style_weight=style_weight
        self.content_weight=content_weight
        self.learning_rate = 0.02
        self.beta_1 = 0.99
        self.epsilon = 1e-1
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)


        self.vgg =  self.vgg_layers(style_layers + content_layers)
        
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        
        self.opt = tf.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, epsilon=self.epsilon)
        self.style_targets={}
        self.content_targets={}
        
    def defineTargets(self,style_image,content_image):
        self.style_targets = self.call(style_image)['style']
        self.content_targets = self.call(content_image)['content']

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                        outputs[self.num_style_layers:])

        style_outputs = [style_output
                        for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}
        
        return {'content':content_dict, 'style':style_dict}
    
    def style_content_loss(self,outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_sum((self.gram_matrix(style_outputs[name])-self.gram_matrix(self.style_targets[name]))**2) /(4* ((self.style_targets[name].numpy().shape[2]*self.style_targets[name].numpy().shape[3])**2 )*self.style_targets[name].numpy().shape[1]**2)
                            for name in style_outputs.keys()])
        style_loss *= (1 - self.content_weight)

        content_loss = tf.add_n([tf.reduce_sum((content_outputs[name]-self.content_targets[name])**2)/(4* self.content_targets[name].numpy().shape[2]*self.content_targets[name].numpy().shape[1]* self.content_targets[name].numpy().shape[3])
                                for name in content_outputs.keys()])
        content_loss *= self.content_weight
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self,image):
        if(len(self.style_targets) != 0  and len(self.content_targets) != 0):
            with tf.GradientTape() as tape:
                outputs = self.call(image)
                loss = self.style_content_loss(outputs)
                grad = tape.gradient(loss, image)
                self.opt.apply_gradients([(grad, image)])
                image.assign(self.clip_0_1(image))