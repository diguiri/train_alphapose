class opt:
  expID = 'default'
  dataset = 'coco'
  nThreads = 1
  snapshot = 1
  addDPG = False
  loadModel = None
  nClasses = 17
  LR = 1e-3
  momentum = 0
  weightDecay = 0
  eps = 1e-8
  crit = 'MSE'
  optMethod = 'rmsprop'
  nEpochs = 3
  epoch = 0
  trainBatch = 1
  validBatch = 1
  trainIters = 3
  valIters = 0
  inputResH = 320
  inputResW = 256
  outputResH = 80
  outputResW = 64
  scale = 0.3
  rotate = 40
  hmGauss = 1

