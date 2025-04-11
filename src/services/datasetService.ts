import axios from 'axios';
import { toast } from "sonner";

const API_BASE_URL ='http://localhost:8000/api';

export interface Dataset {
  id: string;
  name: string;
  size: number;
  created_at: string;
  num_samples: number;
  columns: string[];
  status: 'idle' | 'uploading' | 'validating' | 'success' | 'error';
  error?: string;
}

// Upload new dataset
export const uploadDataset = async (file: File, name: string): Promise<Dataset> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);

    const response = await axios.post(`${API_BASE_URL}/datasets/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    toast.success("Dataset uploaded successfully");
    return {
      ...response.data,
      status: 'success'
    };
  } catch (error) {
    toast.error("Dataset upload failed", {
      description: error.response?.data?.detail || error.message
    });
    throw error;
  }
};

// For compatibility with existing components
export const processDatasetFile = async (file: File): Promise<Dataset> => {
  return uploadDataset(file, file.name.split('.')[0]);
};

// Get all datasets
export const getDatasets = async (): Promise<Dataset[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/datasets`);
    return response.data.map((d: any) => ({
      ...d,
      status: 'success'
    }));
  } catch (error) {
    toast.error("Failed to fetch datasets");
    throw error;
  }
};

// Get single dataset
export const getDataset = async (id: string) => {
  const response = await fetch(`/api/datasets/${id}`);
  if (!response.ok) throw new Error('Dataset not found');
  return response.json();
};

// Delete dataset
export const removeDataset = async (id: string): Promise<void> => {
  try {
    await axios.delete(`${API_BASE_URL}/datasets/${id}`);
    toast.success("Dataset deleted successfully");
  } catch (error) {
    toast.error("Failed to delete dataset");
    throw error;
  }
};

// Clear all datasets (mock implementation)
export const clearDatasets = async (): Promise<void> => {
  toast.warning("Clear datasets not implemented in backend");
};

// Get dataset preview
export const getDatasetPreview = async (id: string, limit = 10): Promise<any[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/datasets/${id}/preview`, {
      params: { limit }
    });
    return response.data;
  } catch (error) {
    toast.error("Failed to fetch dataset preview");
    throw error;
  }
};

export const startTrainingJob = async (
  datasetId: string,
  baseModel: string = "google/gemma-2b-it",
  hyperparameters: {
    learning_rate?: number;
    batch_size?: number;
    num_epochs?: number;
    max_seq_length?: number;
  } = {}
) => {
  try {
    // Verify dataset exists first
    const dataset = await getDataset(datasetId);
    if (!dataset || dataset.status !== 'success') {
      throw new Error("Dataset not ready for training");
    }

    const response = await axios.post(`${API_BASE_URL}/train`, {
      dataset_id: datasetId,
      base_model: baseModel,
      hyperparameters: {
        learning_rate: hyperparameters.learning_rate || 5e-5,
        batch_size: hyperparameters.batch_size || 4,
        num_epochs: hyperparameters.num_epochs || 3,
        max_seq_length: hyperparameters.max_seq_length || 512
      }
    });
    
    return response.data;
  } catch (error: any) {
    console.error("Training error:", error.response?.data || error.message);
    const message = error.response?.data?.detail || 
                   error.response?.data?.message || 
                   error.message;
    throw new Error(message || "Training initialization failed");
  }
};