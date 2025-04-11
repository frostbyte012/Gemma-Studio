
import { DatasetUpload } from "@/components/dataset/DatasetUpload";
import { DatasetPreview } from "@/components/dataset/DatasetPreview";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState, useEffect } from "react";
import { getDatasets } from "@/services/datasetService";
import { toast } from "sonner";

const Datasets = () => {
  const [activeTab, setActiveTab] = useState("upload");
  const [hasUploadedDatasets, setHasUploadedDatasets] = useState(false);
  
  // Check if there are already uploaded datasets
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const datasets = await getDatasets();
        setHasUploadedDatasets(datasets.length > 0);
      } catch (error) {
        console.error("Failed to fetch datasets:", error);
        toast.error("Failed to load datasets");
      }
    };
    fetchDatasets();
  }, []);
  
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Datasets</h1>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-2 md:w-[400px]">
          <TabsTrigger value="upload">Upload</TabsTrigger>
          <TabsTrigger value="preview" disabled={!hasUploadedDatasets}>Preview</TabsTrigger>
        </TabsList>
        
        <TabsContent value="upload" className="space-y-4">
          <DatasetUpload />
        </TabsContent>
        
        <TabsContent value="preview" className="space-y-4">
          <DatasetPreview />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Datasets;
