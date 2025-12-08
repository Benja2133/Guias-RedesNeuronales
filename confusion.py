from sklearn.metrics import confusion_matrix
import numpy as np

# Generar y mostrar matriz de confusión (conteos y normalizada) usando el valid_loader y el modelo ya definidos

# recolectar predicciones y etiquetas verdaderas
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for X, y in valid_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        preds = out.argmax(dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# matriz de confusión (conteos)
cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

# plot de la matriz de confusión (conteos)
fig, ax = plt.subplots(figsize=(8,6))
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(cax, ax=ax)
classes = [labels_name[i] for i in range(10)]
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix (counts)')
# Anotar celdas
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()

# matriz de confusión normalizada por fila (recall por clase)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

fig, ax = plt.subplots(figsize=(8,6))
cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(cax, ax=ax)
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix (normalized by true class)')
# Anotar celdas con porcentaje
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black")
plt.tight_layout()
plt.show()
