import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { useAnimateOnMount } from "@/lib/animations";
import { ArrowRight, Database, BarChart2, HardDrive, Package } from "lucide-react";
import { Gemma } from '@lobehub/icons';

const Index = () => {
  const titleAnimation = useAnimateOnMount({
    type: 'scale',
    duration: 800
  });
  
  const subtitleAnimation = useAnimateOnMount({
    type: 'fade',
    duration: 800,
    delay: 200
  });
  
  const buttonAnimation = useAnimateOnMount({
    type: 'slide',
    direction: 'up',
    duration: 500,
    delay: 600
  });
  
  return (
    <div className="min-h-screen flex flex-col">
      <header className="container mx-auto px-4 py-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Gemma.Combine size={45} />
          <span className="font-bold text-xl tracking-tight">Studio</span>
        </div>
        <nav className="hidden md:flex items-center gap-6">
          <Link to="/dashboard" className="text-muted-foreground hover:text-foreground transition-colors">Dashboard</Link>
          <Link to="/datasets" className="text-muted-foreground hover:text-foreground transition-colors">Datasets</Link>
          <Link to="/training" className="text-muted-foreground hover:text-foreground transition-colors">Training</Link>
          <Button asChild>
            <Link to="/dashboard">Get Started</Link>
          </Button>
        </nav>
        <Button asChild className="md:hidden">
          <Link to="/dashboard">Dashboard</Link>
        </Button>
      </header>
      
      <main className="flex-1 flex flex-col">
        <section className="container mx-auto px-4 py-12 md:py-24 text-center flex flex-col items-center">
          <h1 
            className="text-4xl md:text-6xl font-bold tracking-tight max-w-3xl mx-auto"
            style={titleAnimation.styles}
          >
            Fine-tune Gemma Models with Ease
          </h1>
          <p 
            className="mt-6 text-xl text-muted-foreground max-w-2xl mx-auto"
            style={subtitleAnimation.styles}
          >
            A simple, intuitive interface for fine-tuning Gemma language models on your custom datasets. No complex setup required.
          </p>
          <div 
            className="mt-12 flex flex-col sm:flex-row gap-4 items-center justify-center"
            style={buttonAnimation.styles}
          >
            <Button size="lg" asChild className="btn-transition">
              <Link to="/dashboard">
                Get Started <ArrowRight size={16} className="ml-2" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild className="btn-transition">
              <Link to="/dashboard">View Demo</Link>
            </Button>
          </div>
        </section>
        
        <section className="container mx-auto px-4 py-12 md:py-24">
          <h2 className="text-3xl font-bold tracking-tight text-center mb-12">
            Everything You Need in One Place
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: <Database size={24} className="text-blue-500" />,
                title: "Dataset Management",
                description: "Upload, validate, and preprocess your custom datasets for fine-tuning.",
                delay: 0
              },
              {
                icon: <BarChart2 size={24} className="text-amber-500" />,
                title: "Training Visualization",
                description: "Monitor training progress with real-time metrics and visualizations.",
                delay: 100
              },
              {
                icon: <Package size={24} className="text-violet-500" />,
                title: "Hyperparameter Tuning",
                description: "Easily configure and optimize key hyperparameters for better results.",
                delay: 200
              },
              {
                icon: <HardDrive size={24} className="text-emerald-500" />,
                title: "Model Export",
                description: "Export your fine-tuned models in various formats for deployment.",
                delay: 300
              }
            ].map((feature, index) => (
              <div 
                key={index} 
                className="border rounded-lg p-6 bg-glass card-hover"
                style={{
                  ...useAnimateOnMount({
                    type: 'fade',
                    delay: 800 + feature.delay
                  }).styles
                }}
              >
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>
        
        <section className="bg-muted/30">
          <div className="container mx-auto px-4 py-12 md:py-24 text-center">
            <h2 className="text-3xl font-bold tracking-tight mb-6">
              Ready to Fine-tune Your Own Gemma Model?
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-12">
              Get started in minutes with our intuitive interface. No complex setup required.
            </p>
            <Button size="lg" asChild className="btn-transition">
              <Link to="/dashboard">
                Start Fine-tuning <ArrowRight size={16} className="ml-2" />
              </Link>
            </Button>
          </div>
        </section>
      </main>
      
      <footer className="border-t">
        <div className="container mx-auto px-4 py-8 flex flex-col md:flex-row items-center justify-between">
          <div className="flex items-center gap-2 mb-4 md:mb-0">
            <Gemma.Combine size={24} type={'color'} />
            <span className="font-bold tracking-tight">Studio</span>
          </div>
          <div className="text-sm text-muted-foreground">
            © 2025 Google Deepmind. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;