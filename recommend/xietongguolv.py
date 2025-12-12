# deep_recommender_system.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class NCF(nn.Module):
    """神经协同过滤模型（Neural Collaborative Filtering）"""

    def __init__(self, num_users, num_items, embedding_size=64, hidden_layers=[128, 64, 32]):
        super(NCF, self).__init__()

        # 用户和物品嵌入（GMF部分）
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)

        # 用户和物品嵌入（MLP部分）
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)

        # MLP层
        mlp_layers = []
        input_size = embedding_size * 2
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_size = hidden_size

        self.mlp = nn.Sequential(*mlp_layers)

        # 融合层
        self.fusion = nn.Linear(hidden_layers[-1] + embedding_size, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF部分
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf

        # MLP部分
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)

        # 融合
        fusion_input = torch.cat([gmf_output, mlp_output], dim=1)
        output = torch.sigmoid(self.fusion(fusion_input))

        return output.squeeze()


class RatingDataset(Dataset):
    """评分数据集"""

    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class DeepRecommender:
    """深度推荐系统"""

    def __init__(self, num_users, num_items, embedding_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NCF(num_users, num_items, embedding_size).to(self.device)
        self.num_users = num_users
        self.num_items = num_items

    def train_model(self, train_loader, val_loader, epochs=20, lr=0.001):
        """训练模型"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0

            for user_ids, item_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for user_ids, item_ids, ratings in val_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)

                    predictions = self.model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            print(f'Epoch [{epoch + 1}/{epochs}] | '
                  f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_recommender.pth')

    def predict(self, user_id, item_ids):
        """预测用户对物品的评分"""
        self.model.eval()

        user_ids = torch.LongTensor([user_id] * len(item_ids)).to(self.device)
        item_ids = torch.LongTensor(item_ids).to(self.device)

        with torch.no_grad():
            predictions = self.model(user_ids, item_ids)

        return predictions.cpu().numpy()

    def recommend_top_k(self, user_id, k=10, exclude_items=None):
        """为用户推荐Top-K物品"""
        all_items = list(range(self.num_items))

        if exclude_items:
            all_items = [item for item in all_items if item not in exclude_items]

        predictions = self.predict(user_id, all_items)

        top_k_indices = np.argsort(predictions)[-k:][::-1]
        top_k_items = [all_items[i] for i in top_k_indices]
        top_k_scores = predictions[top_k_indices]

        return list(zip(top_k_items, top_k_scores))


def create_sample_data(num_users=1000, num_items=500, num_ratings=50000):
    """创建示例数据"""
    np.random.seed(42)

    user_ids = np.random.randint(0, num_users, num_ratings)
    item_ids = np.random.randint(0, num_items, num_ratings)
    ratings = np.random.randint(1, 6, num_ratings) / 5.0  # 归一化到0-1

    return user_ids, item_ids, ratings


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    num_users, num_items = 1000, 500
    user_ids, item_ids, ratings = create_sample_data(num_users, num_items)

    # 划分训练集和验证集
    train_users, val_users, train_items, val_items, train_ratings, val_ratings = \
        train_test_split(user_ids, item_ids, ratings, test_size=0.2, random_state=42)

    # 创建数据加载器
    train_dataset = RatingDataset(train_users, train_items, train_ratings)
    val_dataset = RatingDataset(val_users, val_items, val_ratings)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 训练模型
    recommender = DeepRecommender(num_users, num_items)
    recommender.train_model(train_loader, val_loader, epochs=20)

    # 为用户推荐
    user_id = 0
    recommendations = recommender.recommend_top_k(user_id, k=10)
    print(f"\nTop 10 recommendations for user {user_id}:")
    for item_id, score in recommendations:
        print(f"Item {item_id}: {score:.4f}")
