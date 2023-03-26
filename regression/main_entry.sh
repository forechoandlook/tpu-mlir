#!/bin/bash
# set -e

# full test (f32/f16/bf16/int8): main_entry.sh all
# basic test (f32/int8): main_entry.sh basic

OUT=$REGRESSION_PATH/regression_out

test_type=$1

if [ x${test_type} != xall ] && [ x${test_type} != xbasic ]; then
  echo "Error: $0 [basic/all]"
  exit 1
fi

source $REGRESSION_PATH/chip.cfg

mkdir -p $OUT
pushd $OUT

run_regression_net() {
  local net=$1
  local chip_name=$2
  local test_type=$3
  echo "======= run_models.sh $net ${chip_name} ${test_type}====="
  $REGRESSION_PATH/run_model.sh $net ${chip_name} ${test_type} >$net_${chip_name}.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net ${chip_name} regression FAILED" >>result.log
    cat $net_${chip_name}.log >>fail.log
    return 1
  else
    echo "$net ${chip_name} regression PASSED" >>result.log
    return 0
  fi
}

export -f run_regression_net

# run_onnx_op() {
#   echo "======= test_onnx.py ====="
#   chip_list=("bm1684x" "cv183x" "bm1684" "bm1686")
#   ERR=0
#   for chip in ${chip_list[@]}; do
#     test_onnx.py --chip $chip > test_onnx_${chip}.log
#     if [[ "$?" -ne 0 ]]; then
#       echo "test_onnx.py --chip ${chip} FAILED" >>result.log
#       cat test_onnx_${chip}.log >>fail.log
#       ERR=1
#     fi
#     echo "test_onnx.py --chip ${chip} PASSED" >>result.log
#   done
#   return $ERR
# }

run_onnx_op() {
  echo "======= test_onnx.py ====="
  chip_test=("bm1684x" "cv183x")
  ERR=0
  SIMPLE=
  if [ x${test_type} == xbasic ]; then
    SIMPLE="--simple"
  fi
  for chip in ${chip_test[@]}; do
    test_onnx.py --chip $chip $SIMPLE > test_onnx_${chip}.log 2>&1 | true
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
      echo "test_onnx.py --chip ${chip} FAILED" >>result.log
      cat test_onnx_${chip}.log >>fail.log
      ERR=1
      if [ x${test_type} == xbasic ]; then
        return $ERR
      fi
    fi
    echo "test_onnx.py --chip ${chip} PASSED" >>result.log
  done
  return $ERR
}

run_tflite_op() {
  echo "======= test_tflite.py ====="
  test_tflite.py --chip bm1684x > test_tflite_bm1684x.log 2>&1 | true
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    echo "test_tflite.py --chip bm1684x FAILED" >>result.log
    cat test_tflite_bm1684x.log >>fail.log
    return 1
  fi
  echo "test_tflite.py --chip bm1684x PASSED" >>result.log
  return 0
}

run_torch_op() {
  SIMPLE=
  if [ x${test_type} == xbasic ]; then
    SIMPLE="--simple"
  fi
  echo "======= test_torch.py ====="
  test_torch.py --chip bm1684x $SIMPLE > test_torch_bm1684x.log 2>&1 | true
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    echo "test_torch.py --chip bm1684x FAILED" >>result.log
    cat test_torch_bm1684x.log >>fail.log
    return 1
  fi
  echo "test_torch.py --chip bm1684x PASSED" >>result.log
  return 0
}

run_tpulang_op() {
  echo "======= test_tpulang.py ====="
  test_tpulang.py --chip bm1684x > test_tpulang_bm1684x.log 2>&1 | true
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    echo "test_tpulang.py --chip bm1684x FAILED" >>result.log
    cat test_tpulang_bm1684x.log >>fail.log
    return 1
  fi
  echo "test_tpulang.py --chip bm1684x PASSED" >>result.log
  return 0
}

run_script_test() {
  echo "======= script test ====="
  $REGRESSION_PATH/script_test/run.sh > script_test.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "script test FAILED" >>result.log
    cat script_test.log >>fail.log
    return 1
  else
    echo "script test PASSED" >>result.log
    return 0
  fi
}

run_all() {
  echo "" >fail.log
  echo "" >result.log
  ERR=0
  time0=`date +%s`
  run_onnx_op
  if [[ "$?" -ne 0 ]]; then
    ERR=1
    if [ x${test_type} == xbasic ]; then
      return $ERR
    fi
  fi
  time1=`date +%s`
  sumTime=$[ $time1 - $time0 ]
  echo "run_onnx: $sumTime seconds"
  run_tflite_op
  if [[ "$?" -ne 0 ]]; then
    ERR=1
    if [ x${test_type} == xbasic ]; then
      return $ERR
    fi
  fi
  time2=`date +%s`
  sumTime=$[ $time2 - $time1 ]
  echo "run_tflite: $sumTime seconds"
  run_torch_op
  if [[ "$?" -ne 0 ]]; then
    ERR=1
    if [ x${test_type} == xbasic ]; then
      return $ERR
    fi
  fi
  time3=`date +%s`
  sumTime=$[ $time3 - $time2 ]
  echo "run_torch: $sumTime seconds"
  run_tpulang_op
  if [[ "$?" -ne 0 ]]; then
    ERR=1
    if [ x${test_type} == xbasic ]; then
      return $ERR
    fi
  fi
  time4=`date +%s`
  sumTime=$[ $time4 - $time3 ]
  echo "run_tpulang: $sumTime seconds"
  run_script_test
  if [[ "$?" -ne 0 ]]; then
    ERR=1
    if [ x${test_type} == xbasic ]; then
      return $ERR
    fi
  fi
  time5=`date +%s`
  sumTime=$[ $time5 - $time4 ]
  echo "run_script: $sumTime seconds"
  for chip in ${chip_support[@]}; do
    time6=`date +%s`
    echo "" >cmd.txt
    declare -n list="${chip}_model_list_${test_type}"
    for net in ${list[@]}; do
      echo "run_regression_net ${net} ${chip} ${test_type}" >>cmd.txt
    done
    cat cmd.txt
    parallel -j8 --delay 5 --joblog job_regression.log <cmd.txt
    if [[ "$?" -ne 0 ]]; then
      ERR=1
      if [ x${test_type} == xbasic ]; then
        return $ERR
      fi
    fi
    time7=`date +%s`
    sumTime=$[ $time7 - $time6 ]
    echo "run models for $chip: $sumTime seconds"
  done
  time8=`date +%s`
  sumTime=$[ $time8 - $time0 ]
  echo "total time: $sumTime seconds"
  return $ERR
}

run_all
if [ "$?" -ne 0 ]; then
  cat fail.log
  cat result.log
  echo run ${test_type} TEST FAILED
else
  cat result.log
  echo run ${test_type} TEST PASSED
fi

popd

exit $ERR
