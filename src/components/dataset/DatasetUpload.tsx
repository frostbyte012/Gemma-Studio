
import { useState, useCallback } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, FileText, FileCog, Check, AlertCircle, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAnimateOnMount } from "@/lib/animations";
import { startTrainingJob,Dataset, processDatasetFile, removeDataset, clearDatasets } from "@/services/datasetService";
import { toast } from "sonner";
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { uploadDataset,getDataset } from "@/services/datasetService";
import { useNavigate } from "react-router-dom";
import React, { useRef } from "react";


export function DatasetUpload() {
  const queryClient = useQueryClient();

  const navigate = useNavigate();

  const { mutateAsync: uploadFile } = useMutation({
    mutationFn: (file: File) => uploadDataset(file, file.name.split('.')[0]), // Now passes both required params
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      toast.success("Dataset uploaded successfully");
    },
    onError: (error: any) => {
      toast.error(error.message || "Failed to upload dataset");
    }
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const [files, setFiles] = useState<Dataset[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const { styles } = useAnimateOnMount({
    type: 'fade',
    duration: 500
  });
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };
  
  const handleFiles = async (newFiles: File[]) => {
    const validFiles = newFiles.filter(file => 
      [".csv", ".json", ".jsonl", ".txt"].some(ext => file.name.endsWith(ext))
    );
  
    try {
      for (const file of validFiles) {
        console.log("Uploading:", file.name, file.size); // Debug
        const response = await uploadDataset(file, file.name.split('.')[0]);
        console.log("Backend response:", response); // Debug
        
        setFiles(prev => [...prev, {
          ...response,
          status: 'success',
          error: undefined
        }]);
      }
    } catch (error) {
      console.error("Upload failed:", error); // Debug
    }
  };

  const handleStartTraining = async () => {
    try {
      const validDataset = files.find(f => f.status === "success");
      if (!validDataset) {
        toast.error("No valid dataset available");
        return;
      }
  
      toast.loading("Preparing training...");
      
      // Enhanced getDataset with better error handling
      let currentDataset;
      try {
        const response = await fetch(`/api/datasets/${validDataset.id}`);
        const text = await response.text();
        
        // First check if response is HTML error
        if (text.startsWith("<!DOCTYPE") || !response.ok) {
          throw new Error(`Backend returned: ${text.substring(0, 100)}...`);
        }
        
        currentDataset = JSON.parse(text);
      } catch (error) {
        console.error("Dataset fetch error:", error);
        toast.dismiss();
        toast.error("Failed to fetch dataset. Check server logs.");
        return;
      }
  
      if (currentDataset.status !== 'success') {
        toast.dismiss();
        toast.error("Dataset validation failed");
        return;
      }
  
      // Enhanced training job start
      try {
        const result = await startTrainingJob(validDataset.id);
        toast.dismiss();
        toast.success(`Training started successfully`);
        
        navigate('/training', { 
          state: { 
            activeTab: "progress",
            jobId: result.job_id 
          } 
        });
      } catch (error: any) {
        toast.dismiss();
        const errorMsg = error.message.includes("<!DOCTYPE") 
          ? "Server returned an error page" 
          : error.message;
        toast.error(`Failed to start training: ${errorMsg}`);
        console.error("Training error details:", error);
        navigate('/training');
      }
  
    } catch (error: any) {
      toast.dismiss();
      toast.error(error.message || "Unexpected training error");
      console.error("Global training error:", error);
      navigate('/training');
    }
  };
  
  const removeFile = useCallback((id: string) => {
    removeDataset(id);
    setFiles(prev => prev.filter(item => item.id !== id));
  }, []);
  
  const clearAll = useCallback(() => {
    clearDatasets();
    setFiles([]);
  }, []);
  
  const getFileIcon = (fileName: string) => {
    const extension = fileName.substring(fileName.lastIndexOf(".")).toLowerCase();
    
    if (extension.includes(".csv")) {
      return <FileText size={24} className="text-emerald-500" />;
    } else if (extension.includes(".json") || extension.includes(".jsonl")) {
      return <FileCog size={24} className="text-blue-500" />;
    } else {
      return <FileText size={24} className="text-gray-500" />;
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "uploading":
        return <div className="h-5 w-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />;
      case "validating":
        return <div className="h-5 w-5 rounded-full border-2 border-amber-500 border-t-transparent animate-spin" />;
      case "success":
        return <Check size={20} className="text-emerald-500" />;
      case "error":
        return <AlertCircle size={20} className="text-destructive" />;
      default:
        return null;
    }
  };
  
  return (
    <Card className="border shadow-sm" style={styles}>
      <CardHeader>
        <CardTitle>Upload Dataset</CardTitle>
        <CardDescription>
          Upload your datasets for fine-tuning. We support CSV, JSONL, and TXT files.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
                <div 
          className={cn(
            "border-2 border-dashed rounded-lg p-8 text-center",
            isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/20",
            "transition-all duration-200"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center gap-3">
            <Upload size={36} className={cn(
              "text-muted-foreground",
              isDragging ? "text-primary animate-pulse" : ""
            )} />
            <h3 className="text-lg font-medium">Drag & Drop Files</h3>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              Supported formats: CSV, JSONL, and TXT files. Files will be automatically validated for format compatibility.
            </p>
            <div className="mt-4">
              <Button 
                variant="outline" 
                disabled={isProcessing}
                className="btn-transition"
                onClick={handleButtonClick} // Use the new click handler
              >
                Select Files
              </Button>
              <input
                ref={fileInputRef} // Add the ref here
                type="file"
                multiple
                accept=".csv,.jsonl,.json,.txt"
                className="hidden"
                onChange={handleFileChange}
                disabled={isProcessing}
              />
            </div>
          </div>
        </div>
        
        {files.length > 0 && (
          <div className="border rounded-lg divide-y">
            {files.map((item) => (
              <div key={item.id} className="flex items-center gap-3 p-3">
                {getFileIcon(item.name)}
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{item.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(item.size / 1024).toFixed(2)} KB • 
                    {item.status === "error" ? (
                      <span className="text-destructive ml-1">{item.error}</span>
                    ) : (
                      <span className="capitalize ml-1">{item.status}</span>
                    )}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(item.status)}
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    onClick={() => removeFile(item.id)}
                    className="h-7 w-7"
                  >
                    <X size={16} />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="ghost" onClick={clearAll} disabled={files.length === 0}>Clear All</Button>
        <Button 
          disabled={files.length === 0 || !files.some(f => f.status === "success")}
          onClick={handleStartTraining}
        >
          Continue to Training
        </Button>
      </CardFooter>
    </Card>
  );
}
