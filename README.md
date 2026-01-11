# 👁️ Gym Vision Feedback Loop: Observability & Pose Estimation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11_Pose-purple?style=flat&logo=ultralytics&logoColor=white)
![Go](https://img.shields.io/badge/API_Gateway-Golang-00ADD8?style=flat&logo=go&logoColor=white)
![Prometheus](https://img.shields.io/badge/Metrics-Prometheus-E6522C?style=flat&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Dashboard-Grafana-F46800?style=flat&logo=grafana&logoColor=white)


Download dos pesos utilizados pelos modelos (.pt, .onnx) para o KNN e o YOLO

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Modelos-yellow)](https://huggingface.co/daviemmerick/gym-vision-pose-estimation/tree/main)

## 📄 Sobre o Projeto

O **Gym Vision Feedback Loop** é uma plataforma de engenharia de visão computacional projetada para **ambientes de alta performance**.

O grande diferencial deste projeto não é apenas detectar a qualidade de exercícios com **YOLOv11 e KNN**, mas garantir a **Observabilidade Total** do pipeline de inferência. Diferente de implementações acadêmicas, esta arquitetura foi desenhada para produção, monitorando latência, vazão (throughput) e saúde do modelo em tempo real.

### 🎯 Pilares do Projeto

1.  **Engenharia Híbrida (Go + Python):** Combina a concorrência massiva do Golang para gerenciar requisições com a robustez do ecossistema Python para Deep Learning.
2.  **Full Observability:** Integração nativa com **Prometheus e Grafana** para monitorar métricas de sistema (CPU/RAM) e métricas de negócio (Inferências/seg, Drift de Confiança).
3.  **Data-Centric AI:** Pipeline próprio de *Auto-Labeling* para criar datasets proprietários de academia.

## 🛠️ Arquitetura da Solução

O sistema utiliza o padrão **Producer-Consumer** monitorado:

### 1. API Gateway (Golang)

* **Função:** Recebe streams de vídeo e gerencia conexões gRPC.
* **Observabilidade:** Expõe métricas de *Request Rate*, *Latency* e tamanho da fila de processamento (Lag).
* **Concorrência:** Usa *Goroutines* para empilhar tarefas sem bloquear a thread principal.

### 2. Inference Worker & Consumer (Python)

* **Função:** Consome tarefas da fila, executa o YOLOv11 (GPU/CPU) e extrai os keypoints.
* **Observabilidade:** Monitora o tempo de inferência por frame e a confiança média das detecções (para alertar se o modelo começar a "alucinar").

### 3. Monitoring Stack (The Watchtower)

* **Prometheus:** Coleta (scrape) métricas expostas pelo Go e pelo Python a cada 5 segundos.
* **Grafana:** Dashboards visuais para acompanhar a saúde da aplicação em tempo real.

## 📂 Estrutura do Projeto

```text
gym-observability/
├── gym-vision-feedback-loop/   # Inference Worker (Python)
│   ├── train_data/             # Dataset
│   ├── models/                 # Pesos YOLO (best.pt)
│   ├── inference_service.py    # Roda modelo + Prometheus Client
│   └── ...
├── go-api-gateway/             # API Gateway (Golang)
│   ├── main.go                 # Servidor + Prometheus Exporter
│   └── metrics/                # Definição de métricas customizadas
├── monitoring/                 # Infra de Observabilidade
│   ├── prometheus.yml          # Config do Scraper
│   ├── grafana-dashboards/     # JSON dos painéis
│   └── docker-compose.yml      # Sobe Prometheus + Grafana
└── README.md 
```

## 🗺️ Roadmap de Evolução

[x] Data Engineering: Pipeline de ingestão e limpeza.

[x] Auto-Labeling: Rotulagem automática com YOLO.

[X] YOLO Training: Fine-tuning supervisionado.

[X] KNN Training: Análise da forma das poses.

[ ] Golang API: Implementar servidor que recebe vídeo e chama a inferência .

[ ] Integration: Conectar Go e Python via padrão Producer-Consumer.

[ ] Observabilidade: Implementar Prometheus e Grafana para acompanhar a performance dos modelos.


## Resultados

_Em breve_

## Autor

_Davi França Emmerick_
