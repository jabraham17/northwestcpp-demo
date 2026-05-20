use GpuDiagnostics;
use Random;
use Math;
use BlockDist;

proc sigmoid(arr: [?dom] ?eltType) {
  /* Declare a result array with the same domain (and distribution) and eltType */
  var result: arr.type;

  /* Move computation to a GPU sublocale */
  on here.gpus[0] {
    /* Copy the data from CPU to GPU */
    var gpuArr = arr;

    /* Declare a GPU array and compute */
    var gpuResult: [dom] eltType;
    sigmoidHelper(gpuResult, gpuArr);

    /* Copy the result back from GPU to CPU */
    result = gpuResult;
  }
  return result;
}

/* `Xi = 1 / (1 + exp(-Xi))` */
inline proc sigmoidHelper(ref result, arr) {
  result = 1 / (1 + exp(-arr));
}

config const n = 100;
config const print = false;
config const printForGraph = false;
config const verboseGpu = false;
proc main() {
  var Arr: [0..<n] real;

  fillRandom(Arr, -6, 6);

  if verboseGpu then startVerboseGpu();
  var Result = sigmoid(Arr);
  if verboseGpu then stopVerboseGpu();

  if print then writeln(Result);
  if printForGraph {
    for i in 0..<n {
      writeln(Arr[i], " ", Result[i]);
    }
  }
}
