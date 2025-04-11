
import { ModelExport } from "@/components/models/ModelExport";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useEffect, useState } from "react";
import { getLatestTrainingJob } from "@/services/trainingService";

const Models = () => {
  const [hasCompletedTraining, setHasCompletedTraining] = useState(false);

  useEffect(() => {
    // Check if there's a completed training job
    const latestJob = getLatestTrainingJob();
    if (latestJob && latestJob.status === "completed") {
      setHasCompletedTraining(true);
    }
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Models</h1>
      
      {!hasCompletedTraining ? (
        <Card>
          <CardHeader>
            <CardTitle>No Models Available</CardTitle>
            <CardDescription>
              You need to complete a training job before you can export a model
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              To create a model, you need to:
            </p>
            <ol className="list-decimal list-inside space-y-2 mb-4">
              <li>Upload and validate a dataset</li>
              <li>Configure training parameters</li>
              <li>Train a model to completion</li>
            </ol>
            <div className="flex justify-end">
              <button 
                className="bg-primary text-primary-foreground px-4 py-2 rounded hover:bg-primary/90"
                onClick={() => window.location.href = "/datasets"}
              >
                Start with Datasets
              </button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <ModelExport />
      )}
    </div>
  );
};

export default Models;
