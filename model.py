import numpy as np
from sklearn.mixture import GaussianMixture

# CV에 사용할 데이터 나누기
def data_merge(X, target, test_idx, length, k):
    n_mfcc = len(X[0][0][0])
    X_train = np.delete(X, test_idx, axis=1).reshape(10,length*(k-1),n_mfcc)
    y_train = np.delete(target, test_idx, axis=1).reshape(10,length*3, 1)
    X_test = X[:,test_idx, :, :]
    y_test = target[:,test_idx,:,:]
    
    return [X_train, y_train, X_test, y_test]

# Cross Validation
def k_fold_cross_validation(X, target, k):
    acc = 0
    length =  len(X[0][0])
    for test_idx in range(k):
        print(f"K-fold Cross Validation : {test_idx+1}/{k}")
        X_train, y_train, X_test, y_test = data_merge(X, target, test_idx, length, k)
        
        # 혼동 행렬 선언
        confusion_matrices = np.zeros((10, 10), dtype=int)

        # GMM 10개 선언
        models = [GaussianMixture(n_components=5, max_iter=200, covariance_type='tied',
                                                 random_state=1234) for _ in range(10)]
        for i, model in enumerate(models):
            model.fit(X_train[i])

        # 테스트 데이터 합치기
        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test).ravel()
        
        # 예측 및 혼동 행렬 채우기
        for X_t, y_t in zip(X_test, y_test):
            scores = [model.score(X_t.reshape(1,-1)) for model in models]
            predict = np.argmax(scores)
            confusion_matrices[y_t][predict] += 1       
        
        # 정확도 및 혼동행렬 출력
        accuracy = np.trace(confusion_matrices) / np.sum(confusion_matrices)
        print(*confusion_matrices, sep='\n')
        print(f"{test_idx+1}/{k} Accuracy : {accuracy}")
        print("================================")
        acc += accuracy

    # 평균 정확도 반환
    return acc/k