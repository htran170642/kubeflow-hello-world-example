import kfp.dsl as dsl
import kfp.components as components

@dsl.pipeline(
   name='mnist pipeline',
   description='A pipeline to train a model on mnist dataset and start a tensorboard.'
)
def mnist_pipeline(
   path,
   output_path,
   ):
   # import os
   train_op = components.load_component_from_file('./train/component.yaml')
   train_step = train_op(path=path)

   visualize_op = components.load_component_from_file('./tensorboard/component.yaml')
   visualize_step = visualize_op(
     logdir='%s' % train_step.outputs['logdir'],
     output_path=output_path
   )

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   args = parser.parse_args()
   
   import kfp.compiler as compiler
   compiler.Compiler().compile(mnist_pipeline, __file__ + '.zip')