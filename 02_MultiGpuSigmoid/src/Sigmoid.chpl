use GpuDiagnostics;
use Random;
use Math;
use BlockDist;

proc sigmoid(arr: [?dom] ?eltType) {
  import RangeChunk.chunks;
  var result: arr.type;

  /* For each target locale
      compute a chunk of data and move computation to that locale
  */
  const tgtLoc = here.gpus;
  coforall (loc, c) in zip(tgtLoc, chunks(dom.dim(0), tgtLoc.size)) do on loc {
    /* Determine the indices the current locale should use based on the chunk */
    const localGpuDom = dom[c];
  
    /* Copy the data from CPU to GPU */
    var gpuArr = arr[localGpuDom];

    /* Declare a GPU array and compute */
    var gpuResult: [localGpuDom] eltType;
    sigmoidHelper(gpuResult, gpuArr);

    /* Copy the result back from GPU to CPU */
    result[localGpuDom] = gpuResult;
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
