from torch.nn import Module, Embedding, BCEWithLogitsLoss


class BiasedSVD(Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)
        self.user_bias = Embedding(num_users, 1)
        self.item_bias = Embedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()

        dot_product = (user_embeds * item_embeds).sum(dim=1)
        prediction = dot_product + user_b + item_b

        return prediction
