class Inference:
    def __init__(self, model, device, batch_size, num_workers):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def infer(self, data_loader):
        self.model.eval()
        result = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                result.append(outputs)
        return result

    def infer_from_image_folder(self, image_folder):
        data_loader = self._create_data_loader(image_folder)
        return self.infer(data_loader)

    def _create_data_loader(self, image_folder):
        dataset = ImageFolder(image_folder, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)