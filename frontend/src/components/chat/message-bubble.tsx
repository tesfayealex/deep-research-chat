import React from 'react';
// Import types from chat-layout where they are now defined and exported
import { Message, StepResult } from './chat-layout'; 
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Bot, User, Search, Eye, Link as LinkIcon } from 'lucide-react'; // Icons for sender and search steps, and LinkIcon
import ReactMarkdown from 'react-markdown'; // For rendering final report
import remarkGfm from 'remark-gfm'; // GF Markdown support

interface MessageBubbleProps {
  message: Message;
}

// --- Helper Rendering Functions --- 

// Renders the Plan (used in streaming and final view)
const RenderPlanContent: React.FC<{ plan: Message['plan'] }> = ({ plan }) => {
  if (!plan) return null;
  
  // Ensure plan is an array before mapping
  const planArray = Array.isArray(plan) ? plan : [];
  
  if (planArray.length === 0) return null;
  
  return (
    <div className="space-y-1.5 mt-2">
      <h4 className="font-semibold text-foreground text-xs">Research Plan</h4>
      <ul className="list-none space-y-1.5 pl-1 text-muted-foreground">
        {planArray.map((step, index) => (
          <li key={index} className="flex items-start gap-2">
            <span className="font-medium text-foreground w-5 text-right">{index + 1}.</span>
            <div>
              <span className="font-medium text-foreground break-words">{step.step_name || `Step ${index + 1}`}</span>
              <p className="text-muted-foreground break-words">{step.step_detail}</p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

// Renders Executed Steps (used in streaming and final view)
const RenderStepsContent: React.FC<{ steps: StepResult[] }> = ({ steps }) => {
   if (!steps || steps.length === 0) return null;

   // Helper function to check if a string is a valid URL
   const isValidUrl = (string: string): boolean => {
     try {
       new URL(string);
       return true;
     } catch (_) {
       return false;  
     }
   };

   return (
      <div className="space-y-1.5 mt-2">
          <h4 className="font-semibold text-foreground text-xs">Research Steps Executed</h4>
          <div className="space-y-3 pl-1">
            {steps.map((step: StepResult, index: number) => (
              <div key={step.step_index ?? index} className="border-l-2 border-border pl-3 py-1 space-y-1">
                <div className="flex items-center text-xs font-medium text-foreground">
                  <Search className="h-3 w-3 mr-1.5 flex-shrink-0" />
                  <span>{step.step_name || `Step ${step.step_index + 1}`}</span>
                </div>
                {step.findings_preview && (
                  <p className="text-muted-foreground text-xs break-words">
                    {step.findings_preview}
                  </p>
                )}
                {step.sources && step.sources.length > 0 && (
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 pt-1">
                    <span className="text-xs font-medium text-foreground">Sources:</span>
                    {step.sources.map((source, srcIndex) => {
                      const url = typeof source === 'string' ? source : source.original_url;
                      const label = typeof source === 'string' ? new URL(url).hostname : source.label;

                      if (isValidUrl(url)) {
                        // If it's a valid URL, render the link
                        return (
                          <a
                            key={srcIndex}
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-500 underline truncate max-w-[200px]"
                          >
                            <LinkIcon size={12} />
                            <span>{label || new URL(url).hostname}</span>
                          </a>
                        );
                      } else {
                        // Fallback for any non-URL string data
                        return (
                          <span
                            key={srcIndex}
                            className="text-xs text-muted-foreground truncate max-w-[200px]"
                            title={String(source)} // Show full text on hover
                          >
                            {String(source)}
                          </span>
                        );
                      }
                    })}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
    );
};

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isAgent = message.sender === 'agent';

  // Basic styling - refined for Vercel look
  const bubbleClasses = isAgent
    ? 'bg-card dark:bg-zinc-800 rounded-lg p-3 max-w-[85%] w-fit self-start flex gap-2' // Agent messages align left
    : 'bg-primary text-primary-foreground rounded-lg p-3 max-w-[85%] w-fit self-end flex gap-2'; // User messages align right

  const icon = isAgent 
    ? <Bot className="h-5 w-5 text-muted-foreground flex-shrink-0 mt-1" /> 
    : <User className="h-5 w-5 flex-shrink-0 mt-1" />;

  // Enhanced renderThinking function for FINAL display
  const renderThinking = () => {
    const hasPlan = message.plan && message.plan.length > 0;
    const hasThinking = message.thinking && message.thinking.length > 0;
    if (!hasPlan && !hasThinking) return null;
    
    return (
      <Accordion type="single" collapsible className="w-full text-xs mb-3 border-b border-border pb-3">
        <AccordionItem value="thinking" className="border-0">
          <AccordionTrigger className="text-muted-foreground hover:no-underline py-1 px-0 font-normal flex gap-1.5 items-center">
            <Eye size={14} />
            <span>Show reasoning</span>
          </AccordionTrigger>
          <AccordionContent className="pt-3 pb-0 px-0">
            <div className="space-y-4">
              {/* Use helper for Plan */} 
              <RenderPlanContent plan={message.plan} />
              {/* Use helper for Steps */} 
              <RenderStepsContent steps={message.thinking || []} /> 
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    );
  };

  return (
    <div className={bubbleClasses}>
      {icon}
      <div className="flex-grow min-w-0 space-y-1"> 
        {isAgent && message.isStreaming ? (
          <div className="w-full">
            <div className="flex items-center text-xs text-muted-foreground mb-2">
              <span>{message.status || 'Thinking...'}</span>
              {message.status !== 'Finished' && message.status !== 'Error' && (
                <div className="ml-2 h-3 w-3 animate-spin rounded-full border-2 border-solid border-current border-r-transparent" role="status" />
              )}
            </div>
            
            {/* Render intermediate text (e.g., "Okay here's the plan") */}
            {message.text && (
              <div className="prose prose-sm dark:prose-invert max-w-none break-words mb-2">
                 {/* We might not need markdown here if it's just simple text now */}
                <p>{message.text}</p> 
              </div>
            )}

            {/* We're removing this redundant plan display from the streaming UI, it's still in thinking */}
            {/* The plan will only be visible within the thinking accordion */}

            {/* Render accumulating steps during streaming using helper */} 
            <RenderStepsContent steps={message.thinking || []} /> 

          </div>
        ) : (
          // Render final message
          <>
            {/* Final Collapsible Reasoning uses helpers */} 
            {renderThinking()} 
            
            {/* Final Report Markdown Content */} 
            <div className="prose prose-sm dark:prose-invert max-w-none break-words">
              {typeof message.text === 'string' && (
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]} // Ensure GFM is included
                  // Ensure link component override is present for final render
                  components={{
                    a: ({node, ...props}) => 
                      <a {...props} target="_blank" rel="noopener noreferrer" className="text-primary underline hover:text-primary/90" />
                  }}
                >
                  {message.text}
                </ReactMarkdown>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
} 