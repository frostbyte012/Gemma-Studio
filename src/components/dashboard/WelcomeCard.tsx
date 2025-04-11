
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { useAnimateOnMount } from "@/lib/animations";

export function WelcomeCard() {
  const { styles } = useAnimateOnMount({
    type: 'slide',
    direction: 'up',
    duration: 500
  });
  
  return (
    <Card className="border-none shadow-lg bg-glass" style={styles}>
      <CardHeader>
        <CardTitle className="text-2xl font-bold tracking-tight">Welcome to Gemma Studio</CardTitle>
        <CardDescription>
          Fine-tune Gemma models on your own datasets with an intuitive interface
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-muted-foreground">
          Easily upload your data, configure training parameters, and export your fine-tuned models - all through a simple, streamlined interface.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 pt-2">
          <Button asChild className="btn-transition">
            <Link to="/datasets">
              Upload Dataset <ArrowRight size={16} className="ml-2" />
            </Link>
          </Button>
          <Button variant="outline" asChild className="btn-transition">
            <Link to="/training">
              Start Training
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
