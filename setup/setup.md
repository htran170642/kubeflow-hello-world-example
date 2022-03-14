minikube
```
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```
kubectl
```
curl -LO https://dl.k8s.io/release/v1.19.0/bin/linux/amd64/kubectl
sudo install kubectl /usr/local/bin/kubectl
```
start minikube cluster
```
minikube start --kubernetes-version=v1.19.0 --cpus 16 --memory 12g --disk-size 48g --kvm-gpu \
    --extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/sa.key \
    --extra-config=apiserver.service-account-key-file=/var/lib/minikube/certs/sa.pub \
    --extra-config=apiserver.service-account-issuer=api \
    --extra-config=apiserver.service-account-api-audiences=api,spire-server,nats \
    --extra-config=apiserver.authorization-mode=Node,RBAC \
    --extra-config=kubelet.authentication-token-webhook=true 
```

install kubeflow <br>
github link: https://github.com/kubeflow/manifests/tree/v1.4.0 <br>
prerequisites https://github.com/kubeflow/manifests/tree/v1.4.0#prerequisites <br>
command https://github.com/kubeflow/manifests/tree/v1.4.0#install-with-a-single-command <br>
