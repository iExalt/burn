use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_matmul() {
    let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[11.0, 5.0], [11.0, 5.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[3.0, 3.0], [10.0, 10.0]]), false);
    tensor_3
        .to_data()
        .assert_eq(&TensorData::from([[18.0, 28.0], [14.0, 23.0]]), false);
}

#[test]
fn test_matmul_complex_1() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_5 = tensor_4.matmul(tensor_3);

    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[44.0, 20.0], [44.0, 20.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[56.0, 56.0], [16.0, 16.0]]), false);
}

#[test]
fn test_matmul_complex_2() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_5 = tensor_4.matmul(tensor_3.clone());
    let tensor_6 = tensor_1.clone().matmul(tensor_5);

    let grads = tensor_6.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[800.0, 792.0], [360.0, 592.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[264., 264.0], [344.0, 344.0]]), false);
}

#[cfg(feature = "fusion")]
#[test]
fn test_matmul_terminalo3_branched_linear_x_backward() {
    let device = AutodiffDevice::new();
    let input = TestTensor::<2>::ones([2048, 512], &device).require_grad();
    let state_weight = TestTensor::<2>::ones([512, 256], &device).require_grad();
    let value_weight = TestTensor::<2>::ones([512, 1], &device).require_grad();

    let state = input.clone().matmul(state_weight).sum();
    let value = input.clone().matmul(value_weight).sum();
    let grads = state.add(value).backward();
    let input_grad = input.grad(&grads).unwrap();
    let expected = TensorData::new(vec![257.0f32; 2048 * 512], [2048, 512]);

    input_grad
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
