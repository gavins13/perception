import unittest
import sys
sys.path.insert(0, 'capsules/operations')
from capsule_operations import LocalisedCapsuleLayer

class LocalisedCapsuleLayerTest(unittest.TestCase):
    def setUp(self):
        self.M = 120
        self.input = {"channels": 9, "vec_dim": 8, "x": 128, "y": 128}
        self.output = {"channels": 5, "vec_dim": 4}
        self.k = 3
        self.CapsLayer = LocalisedCapsuleLayer(k=self.k, output_vec_dim=self.output["vec_dim"], num_output_channels=self.output["channels"])
    def test_transform(self):
        input_tensor = tf.placeholder(tf.float32, shape=[self.M, self.input["vec_dim"], self.input["channels"], self.input["x"],self.input["y"]])
        output_tensor = self.CapsLayer.transform(input_tensor)
        self.assertItemsEqual(output_tensor.get_shape().as_list(), [self.M,self.input["x"], self.input["y"], self.output["vec_dim"], self.input["channels"], self.output["channels"]])
    def test_localise(self):
        input_tensor = tf.placeholder(tf.float32, shape=[self.M,  self.input["x"],self.input["y"], self.output["vec_dim"], self.input["channels"], self.output["channels"]])
        output_tensor = self.CapsLayer.localise(input_tensor)
        self.assertItemsEqual(output_tensor.get_shape().as_list(), [self.M,  self.output["vec_dim"],  self.output["channels"], self.input["x"]*self.input["y"], self.k*self.k*self.input["channels"]])
    def test_route(self):
        input_tensor = [self.M,  self.output["vec_dim"],  self.output["channels"], self.input["x"]*self.input["y"], self.k*self.k*self.input["channels"]]
        output_tensor = self.CapsLayer.route(input_tensor)
        self.assertItemsEqual(output_tensor.get_shape().as_list(), [self.M,  self.output["vec_dim"],  self.output["channels"], self.input["x"],self.input["y"]])
    def test_all(self):
        input_tensor = tf.placeholder(tf.float32, shape=[self.M, self.input["vec_dim"], self.input["channels"], self.input["x"],self.input["y"]])
        output_tensor = self.CapsLayer(input_tensor, "TestLocalCaps")
        self.assertItemsEqual(output_tensor.get_shape().as_list(), [self.M,  self.output["vec_dim"],  self.output["channels"], self.input["x"],self.input["y"]])



if(__name__ == "main"):
    unittest.main()
