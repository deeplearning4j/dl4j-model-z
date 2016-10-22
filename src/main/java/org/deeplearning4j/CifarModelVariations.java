package org.deeplearning4j;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *Cifar10 Caffe Model Variations
 *
 * Quick Model based on:
 * http://caffe.berkeleyvision.org/gathered/examples/cifar10.html
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_train_test.prototxt
 *
 * Full Sigmoid Model based on:
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_solver_lr1.prototxt
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt

 *
 * BatchNorm Model based on:
 *  https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_sigmoid_solver_bn.prototxt
 *  https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt
 */
public class CifarModelVariations {

    private int height;
    private int width;
    private int channels;
    private int numLabels;
    private long seed;
    private int iterations;
    protected int[] nIn;
    protected int[] nOut;
    protected String activation;
    protected WeightInit weightInit;
    protected OptimizationAlgorithm optimizationAlgorithm;
    protected Updater updater;
    protected LossFunctions.LossFunction lossFunctions;
    protected double learningRate;
    protected double biasLearningRate;
    protected boolean regularization;
    protected double l2;
    protected double momentum;

    MultiLayerConfiguration conf;


    public CifarModelVariations(int height, int width, int channels, int numLabels, long seed,
                       int iterations, int[] nIn, int[] nOut, String activation,
                       WeightInit weightInit, OptimizationAlgorithm optimizationAlgorithm,
                       Updater updater, LossFunctions.LossFunction lossFunctions,
                       double learningRate, double biasLearningRate,
                       boolean regularization, double l2, double momentum) {

        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
        this.optimizationAlgorithm = optimizationAlgorithm;
        this.updater = updater;
        this.lossFunctions = lossFunctions;
        this.learningRate = learningRate;
        this.biasLearningRate = (biasLearningRate == Double.NaN)? 0.0: biasLearningRate;
        this.regularization = regularization;
        this.l2 = l2;
        this.momentum = momentum;
    }

    private ConvolutionLayer conv1x1act(String name, int out, double std, int[] padding, String activation) {
        return new ConvolutionLayer.Builder(new int[]{1,1}, new int[]{1,1}, padding).name(name).nOut(out).activation(activation).dist(new GaussianDistribution(0, std)).build();
    }

    private ConvolutionLayer conv3x3act(String name, int nIn, int out) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1}).name(name).nIn(nIn).nOut(out).activation("identity").build();
    }

    private ConvolutionLayer conv3x3dropact(String name, int nIn, int out, double dropOut) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1}).name(name).nIn(nIn).nOut(out).activation("identity").dropOut(dropOut).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, double std, int[] padding) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, padding).name(name).nOut(out).dist(new GaussianDistribution(0, std)).build();
    }

    private ConvolutionLayer conv5x5nin(String name, int nIn, int out, double std, int[] padding) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, padding).name(name).nIn(nIn).nOut(out).dist(new GaussianDistribution(0, std)).build();
    }

    private ConvolutionLayer conv5x5act(String name, int nIn, int out, double std, int[] padding, String activation) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, padding).name(name).nIn(nIn).nOut(out).activation(activation).dist(new GaussianDistribution(0, std)).build();
    }

    private ConvolutionLayer conv5x5bias(String name, int nIn, int out, double std, int[] padding, double biasInit) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, padding).name(name).nIn(nIn).nOut(out).dist(new GaussianDistribution(0, std)).biasInit(biasInit).build();
    }

    private ConvolutionLayer conv5x5dropact(String name, int out, double std, int[] padding, String activation) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, padding).name(name).nOut(out).activation(activation).dropOut(0.5).dist(new GaussianDistribution(0, std)).build();
    }

    private SubsamplingLayer maxPool2x2(String name) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2}).name(name).build();
    }

    private SubsamplingLayer maxPool3x3(String name) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2}).name(name).build();
    }

    private SubsamplingLayer maxPool6x6(String name) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{6, 6}, new int[]{1, 1}).name(name).build();
    }

    private SubsamplingLayer avgPool3x3(String name) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3}, new int[]{2, 2}).name(name).build();
    }

    private LocalResponseNormalization lrn(String name, int k, double alpha, double beta, int n) {
        return new LocalResponseNormalization.Builder(k, alpha, beta).name(name).n(n).build();
    }

    private LocalResponseNormalization lrnBias(String name, int k, double alpha, double beta, int n) {
        return new LocalResponseNormalization.Builder(k, alpha, beta).name(name).n(n).biasInit(1).build();
    }

    private DenseLayer dense(String name, int nout, double dropout) {
        return new DenseLayer.Builder().name(name).nOut(nout).dropOut(dropout).build();
    }

    private DenseLayer denseNorm(String name, int nout, double dropout, double std) {
        return new DenseLayer.Builder().name(name).nOut(nout).dropOut(dropout).dist(new GaussianDistribution(0, std)).build();
    }

    private DenseLayer denseL2Bias(String name, int nout, double std, double bias, double l2) {
        return new DenseLayer.Builder().name(name).nOut(nout).l2(l2).biasInit(bias).dist(new NormalDistribution(0, std)).build();
    }

    private OutputLayer output(String name, int nout, double std) {
        return new OutputLayer.Builder(lossFunctions).name(name).activation("softmax").nOut(nout).dist(new GaussianDistribution(0, std)).build();
    }

    public MultiLayerConfiguration caffeInitQuick() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .weightInit(weightInit)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .updater(updater).momentum(momentum)
                .regularization(regularization).l2(l2)
                .list()
                .layer(0, conv5x5nin("cnn1", channels, nOut[0], 1e-4, new int[]{2,2}))
                .layer(1, maxPool3x3("pool1"))
                .layer(2, conv5x5("cnn2", nOut[1], 1e-2, new int[]{2,2}))
                .layer(3, avgPool3x3("pool2"))
                .layer(4, conv5x5("cnn3", nOut[2], 1e-2, new int[]{2,2}))
                .layer(5, avgPool3x3("pool3"))
                .layer(6, denseNorm("ffn1", nOut[3], 0.5, 1e-2))
                .layer(7, output("output", numLabels, 1e-2))
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);
//                .setInputType(InputType.convolutionalFlat(height,width,channels));

        return builder.build();
    }

    public MultiLayerConfiguration caffeInitFull() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .weightInit(weightInit).dist(new GaussianDistribution(0, 1e-4))
                .activation(activation)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .updater(updater).momentum(momentum)
                .regularization(regularization).l2(l2)
                .list()
                .layer(0, conv5x5nin("cnn1", channels, nOut[0], 1e-2, new int[]{2,2})) // TODO double check gaussian
                .layer(1, maxPool3x3("pool1"))
                .layer(2, lrn("lrn1", 1, 5e-05, 0.75, 3))
                .layer(3, conv5x5("cnn2", nOut[1], 1e-2, new int[]{2,2}))
                .layer(4, avgPool3x3("pool2"))
                .layer(5, lrn("lrn1", 1, 5e-05, 0.75, 3))
                .layer(6, conv5x5("cnn3", nOut[2], 1e-2, new int[]{2,2}))
                .layer(7, avgPool3x3("pool3"))
                .layer(8, denseNorm("ffn1", nOut[3], 0.5, 1e-2))
                .layer(9, output("output", numLabels, 1e-2))
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);
        conf = builder.build();
        return conf;
    }

    public MultiLayerConfiguration caffeInitBN() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .weightInit(weightInit).dist(new GaussianDistribution(0, 1e-4))
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(1)
                .lrPolicySteps(5000)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .updater(updater).momentum(momentum)
                .regularization(regularization).l2(l2)
                .list()
                .layer(0, conv5x5act("cnn1", channels, nOut[0], 1e-4, new int[]{2,2},  "identity"))
                .layer(1, maxPool3x3("pool1"))
                .layer(2, new BatchNormalization.Builder().build())
                .layer(3, new ActivationLayer.Builder().build())
                .layer(4, conv5x5act("cnn2", 0, nOut[1], 1e-2, new int[]{2,2}, "identity"))
                .layer(5, new BatchNormalization.Builder().build())
                .layer(6, new ActivationLayer.Builder().build())
                .layer(7, avgPool3x3("pool2"))
                .layer(8, conv5x5act("cnn3", 0, nOut[2], 1e-2, new int[]{2,2}, "identity"))
                .layer(9, new BatchNormalization.Builder().build())
                .layer(10, new ActivationLayer.Builder().build())
                .layer(11, avgPool3x3("pool3"))
                .layer(12, output("output", numLabels, 1e-2))
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);
        conf = builder.build();
        return conf;
    }

    public MultiLayerConfiguration tensorflowInference(){
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(activation)
                .updater(updater)
                .weightInit(weightInit).dist(new NormalDistribution(0, 5e-2))
                .iterations(iterations)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate)
                .regularization(true)
                .l2(l2) // TODO pass in 0
                .momentum(momentum)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(136500) // TF does it every 136,500 iter with 1M cap
                .list()
                .layer(0, conv5x5nin("cnn1", channels, nOut[0], 5e-2, new int[]{0,0}))
                .layer(1, maxPool3x3("pool1"))
                .layer(2, lrnBias("lrn1", 1, 0.001/9.0, 0.75, 4))
                .layer(3, conv5x5bias("cnn2", channels, nOut[0], 5e-2, new int[]{0,0}, 0.1))
                .layer(4, lrnBias("lrn2", 1, 0.001/9.0, 0.75, 4))
                .layer(5, maxPool3x3("pool2"))
                .layer(6, denseL2Bias("ffn1", nOut[1], 4e-2, 0.1, 4e-3))
                .layer(7, denseL2Bias("ffn2", nOut[2], 4e-2, 0.1, 4e-3))
                .layer(8, output("output", numLabels, 1/192.0))
                .backprop(true).pretrain(false)
                .cnnInputSize(height,width,channels);
        conf = builder.build();
        return conf;

    }

    public MultiLayerConfiguration torchInitNin(){

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(activation)
                .updater(updater)
                .weightInit(weightInit).dist(new NormalDistribution(0, 0.5))
                .iterations(iterations)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate)
                .regularization(true).l2(l2)
                .learningRateDecayPolicy(LearningRatePolicy.TorchStep)
                .lrPolicyDecayRate(.5)
                .lrPolicySteps(25)
                .momentum(momentum)
                .list()
                .layer(0, conv5x5act("cnn1", channels, nOut[0], 0.5, new int[]{2,2}, "identity"))
                .layer(1, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(2, new ActivationLayer.Builder().build())
                .layer(3, conv5x5act("cnn2", 0, nOut[1], 0.5, new int[]{0,0}, "identity"))
                .layer(4, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(5, new ActivationLayer.Builder().build())
                .layer(6, conv5x5act("cnn3", 0, nOut[2], 0.5, new int[]{0,0}, "identity"))
                .layer(7, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(8, new ActivationLayer.Builder().build())
                .layer(9, maxPool3x3("pool1"))
                .layer(10, conv5x5dropact("cnn4", 0, nOut[3], new int[]{2,2}, "identity"))
                .layer(11, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(12, new ActivationLayer.Builder().build())
                .layer(13, conv5x5act("cnn5", 0, nOut[4], 0.5, new int[]{0,0}, "identity"))
                .layer(14, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(15, new ActivationLayer.Builder().build())
                .layer(16, conv5x5act("cnn6", 0, nOut[5], 0.5, new int[]{0,0}, "identity"))
                .layer(17, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(18, new ActivationLayer.Builder().build())
                .layer(19, maxPool3x3("pool2")) //TODO 32 not large enough for this setup - verify config - gets to size 3 here
                .layer(20, conv3x3dropact("cnn7", 0, nOut[6], 0.5))
                .layer(21, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(22, new ActivationLayer.Builder().build())
                .layer(23, conv1x1act("cnn8", 0, nOut[7], new int[]{0,0}, "identity"))
                .layer(24, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(25, new ActivationLayer.Builder().build())
                .layer(26, conv1x1act("cnn9", 0, nOut[8], new int[]{1,1}, "identity"))
                .layer(27, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(28, new ActivationLayer.Builder().build())
                .layer(29, maxPool6x6("pool3"))
                .layer(30, output("output", numLabels, 1/192.0))
                .backprop(true).pretrain(false)
                .cnnInputSize(height,width,channels);
        conf = builder.build();
        return conf;


    }

    public MultiLayerConfiguration torchInitVGG() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(activation)
                .updater(updater)
                .weightInit(weightInit)
                .iterations(iterations)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate)
                .regularization(true).l2(l2)
                .learningRateDecayPolicy(LearningRatePolicy.TorchStep)
                .lrPolicyDecayRate(.5)
                .lrPolicySteps(9765) // 9765 = (25*50000)/128 - based on iterations
                // TODO need momentum applied when not nesterovs
                // TODO need lr decay applied in sgd when provided
                // TODO add weight init for caffe for Xavier and recompare lenet
                .momentum(momentum)
                .list()
                .layer(0, conv3x3dropact("cnn1", channels, nOut[0], 0.3))
                .layer(1, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(2, new ActivationLayer.Builder().build())
                .layer(3, conv3x3act("cnn2", 0, nOut[1]))
                .layer(4, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(5, new ActivationLayer.Builder().build())
                .layer(6, maxPool2x2("pool1"))
                .layer(7, conv3x3dropact("cnn3", 0, nOut[2], 0.4))
                .layer(8, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(9, new ActivationLayer.Builder().build())
                .layer(10, conv3x3act("cnn4", 0, nOut[3]))
                .layer(11, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(12, new ActivationLayer.Builder().build())
                .layer(13, maxPool2x2("pool2"))
                .layer(14, conv3x3dropact("cnn5", 0, nOut[4], 0.4))
                .layer(15, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(16, new ActivationLayer.Builder().build())
                .layer(17, conv3x3dropact("cnn6", 0, nOut[5], 0.4))
                .layer(18, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(19, new ActivationLayer.Builder().build())
                .layer(20, conv3x3act("cnn7", 0, nOut[6]))
                .layer(21, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(22, new ActivationLayer.Builder().build())
                .layer(23, maxPool2x2("pool3"))
                .layer(24, conv3x3dropact("cnn8", 0, nOut[7], 0.4))
                .layer(25, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(26, new ActivationLayer.Builder().build())
                .layer(27, conv3x3dropact("cnn9", 0, nOut[8], 0.4))
                .layer(28, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(29, new ActivationLayer.Builder().build())
                .layer(30, conv3x3act("cnn10", 0, nOut[9]))
                .layer(31, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(32, new ActivationLayer.Builder().build())
                .layer(33, maxPool2x2("pool4"))
                .layer(34, conv3x3dropact("cnn11", 0, nOut[10], 0.4))
                .layer(35, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(36, new ActivationLayer.Builder().build())
                .layer(37, conv3x3dropact("cnn12", 0, nOut[11], 0.4))
                .layer(38, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(39, new ActivationLayer.Builder().build())
                .layer(40, conv3x3act("cnn13", 0, nOut[12]))
                .layer(41, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(42, new ActivationLayer.Builder().build())
                .layer(43, maxPool2x2("pool5"))
                .layer(44, dense("ffn1", nOut[13], 0.5))
                .layer(45, new BatchNormalization.Builder().eps(1e-3).build())
                .layer(46, new ActivationLayer.Builder().build())
                .layer(47, dense("ffn2", nOut[14], 0.5))
                .layer(48, output("output", numLabels, 1/192.0))
                .backprop(true).pretrain(false)
                .cnnInputSize(height,width,channels);

        conf = builder.build();
        return conf;

    }

    public MultiLayerNetwork buildNetwork(CifarModeEnum networkType) {
        switch (networkType) {
            case CAFFE_QUICK:
                conf = caffeInitQuick();
                break;
            case CAFFE_FULL_SIGMOID:
                conf = caffeInitFull();
                break;
            case CAFFE_BATCH_NORM:
                conf = caffeInitBN();
                break;
            case TENSORFLOW_INFERENCE:
                conf = tensorflowInference();
                break;
            case TORCH_NIN:
                conf = torchInitNin();
                break;
            case TORCH_VGG:
                conf = torchInitVGG();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");

        }

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }

    public enum CifarModeEnum {
        CAFFE_BATCH_NORM, CAFFE_FULL_SIGMOID, CAFFE_QUICK, TENSORFLOW_INFERENCE, TORCH_NIN, TORCH_VGG;
    }

}

