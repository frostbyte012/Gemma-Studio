
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { useAnimateOnMount } from "@/lib/animations";

// Sample data
const sampleCSVData = [
  { prompt: "Explain how transformers work in NLP", completion: "Transformers are a type of neural network architecture that uses self-attention mechanisms..." },
  { prompt: "What is fine-tuning in machine learning?", completion: "Fine-tuning is the process of taking a pre-trained model and further training it on a specific dataset..." },
  { prompt: "Define what a language model is", completion: "A language model is a type of artificial intelligence model that can understand, interpret and generate human language..." },
  { prompt: "Explain the concept of embedding in NLP", completion: "In natural language processing, embeddings are dense vector representations of words or tokens..." },
  { prompt: "What is a token in NLP?", completion: "In NLP, a token is a basic unit of text that has been segmented for processing. Tokens are typically words, subwords..." },
];

interface DatasetStats {
  totalExamples: number;
  avgPromptLength: number;
  avgCompletionLength: number;
  minPromptLength: number;
  maxPromptLength: number;
  minCompletionLength: number;
  maxCompletionLength: number;
}

const sampleStats: DatasetStats = {
  totalExamples: 1250,
  avgPromptLength: 62,
  avgCompletionLength: 128,
  minPromptLength: 10,
  maxPromptLength: 230,
  minCompletionLength: 15,
  maxCompletionLength: 512,
};

export function DatasetPreview() {
  const [activeTab, setActiveTab] = useState("preview");
  
  const { styles } = useAnimateOnMount({
    type: 'slide',
    direction: 'up',
    duration: 500,
    delay: 200
  });
  
  return (
    <Card className="border shadow-sm" style={styles}>
      <CardHeader>
        <CardTitle>Dataset Preview</CardTitle>
        <CardDescription>
          Review your dataset before fine-tuning
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="preview" onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid grid-cols-3 w-full md:w-[400px]">
            <TabsTrigger value="preview">Preview</TabsTrigger>
            <TabsTrigger value="statistics">Statistics</TabsTrigger>
            <TabsTrigger value="validation">Validation</TabsTrigger>
          </TabsList>
          
          <TabsContent value="preview" className="space-y-4">
            <div className="border rounded-lg overflow-hidden">
              <div className="overflow-auto max-h-[400px]">
                <Table>
                  <TableHeader className="sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                    <TableRow>
                      <TableHead className="w-12">#</TableHead>
                      <TableHead>Prompt</TableHead>
                      <TableHead>Completion</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sampleCSVData.map((row, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{index + 1}</TableCell>
                        <TableCell className="max-w-[200px] truncate" title={row.prompt}>
                          {row.prompt}
                        </TableCell>
                        <TableCell className="max-w-[300px] truncate" title={row.completion}>
                          {row.completion}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
            <div className="flex justify-end">
              <Button size="sm" variant="outline" 
                onClick={() => setActiveTab("statistics")}>
                View Statistics
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="statistics" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <StatCard 
                title="Dataset Overview" 
                items={[
                  { label: "Total Examples", value: sampleStats.totalExamples.toString() },
                  { label: "Avg. Prompt Length", value: `${sampleStats.avgPromptLength} tokens` },
                  { label: "Avg. Completion Length", value: `${sampleStats.avgCompletionLength} tokens` }
                ]} 
              />
              <StatCard 
                title="Length Distribution" 
                items={[
                  { label: "Prompt Length (Min/Max)", value: `${sampleStats.minPromptLength} / ${sampleStats.maxPromptLength}` },
                  { label: "Completion Length (Min/Max)", value: `${sampleStats.minCompletionLength} / ${sampleStats.maxCompletionLength}` }
                ]} 
              />
            </div>
            <div className="flex justify-end space-x-3">
              <Button size="sm" variant="outline" 
                onClick={() => setActiveTab("preview")}>
                Back to Preview
              </Button>
              <Button size="sm" variant="outline" 
                onClick={() => setActiveTab("validation")}>
                View Validation
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="validation" className="space-y-4">
            <div className="border rounded-lg p-4 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300 flex items-center gap-3">
              <div className="h-8 w-8 rounded-full bg-emerald-100 dark:bg-emerald-900 flex items-center justify-center flex-shrink-0">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-check"><path d="M20 6 9 17l-5-5"/></svg>
              </div>
              <div>
                <h3 className="font-medium">Dataset is valid for fine-tuning</h3>
                <p className="text-sm text-emerald-600 dark:text-emerald-400">No issues found in the dataset structure</p>
              </div>
            </div>
            
            <div className="space-y-2">
              <h3 className="font-medium">Validation Checks</h3>
              <div className="space-y-3">
                {[
                  { label: "Format validation", status: "success" },
                  { label: "Schema validation", status: "success" },
                  { label: "Empty fields check", status: "success" },
                  { label: "Token limit check", status: "success" },
                ].map((check, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm">
                    <div className={cn(
                      "h-5 w-5 rounded-full flex items-center justify-center",
                      check.status === "success" ? "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400" : ""
                    )}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-check"><path d="M20 6 9 17l-5-5"/></svg>
                    </div>
                    <span>{check.label}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="flex justify-end">
              <Button size="sm" variant="outline" 
                onClick={() => setActiveTab("statistics")}>
                Back to Statistics
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

interface StatCardProps {
  title: string;
  items: { label: string; value: string }[];
}

function StatCard({ title, items }: StatCardProps) {
  return (
    <div className="border rounded-lg p-4">
      <h3 className="font-medium mb-3">{title}</h3>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item.label} className="flex justify-between text-sm">
            <span className="text-muted-foreground">{item.label}</span>
            <span className="font-medium">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
