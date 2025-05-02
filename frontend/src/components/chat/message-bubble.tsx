import React from 'react';
import { Message } from './chat-layout'; // Assuming Message type is exported from ChatLayout
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Bot, User, Search, Eye, Link as LinkIcon } from 'lucide-react'; // Icons for sender and search steps, and LinkIcon
import ReactMarkdown from 'react-markdown'; // For rendering final report
import remarkGfm from 'remark-gfm'; // GF Markdown support

// Define StepResult type locally for thinking steps received from backend
interface StepResult {
  step_index: number;
  step_name?: string;
  findings_preview?: string;
  sources?: string[];
  // Allow any other potential fields from backend
  [key: string]: any; 
}

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isAgent = message.sender === 'agent';

  // Basic styling - refined for Vercel look
  const bubbleClasses = isAgent
    ? 'bg-card dark:bg-zinc-800 rounded-lg p-3 max-w-[85%] w-fit self-start flex gap-2' // Agent messages align left
    : 'bg-primary text-primary-foreground rounded-lg p-3 max-w-[85%] w-fit self-end flex gap-2'; // User messages align right

  const icon = isAgent 
    ? <Bot className="h-5 w-5 text-muted-foreground flex-shrink-0 mt-1" /> 
    : <User className="h-5 w-5 flex-shrink-0 mt-1" />;

  // Enhanced renderThinking function
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
            <div className="space-y-4"> {/* Increased spacing between sections */}
              {/* Display Plan */}
              {hasPlan && (
                <div className="space-y-1.5">
                  <h4 className="font-semibold text-foreground text-xs">Research Plan</h4>
                  <ul className="list-none space-y-1.5 pl-1 text-muted-foreground">
                    {message.plan!.map((step, index) => (
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
              )}
              
              {/* Display Executed Steps (Thinking) */} 
              {hasThinking && (
                <div className="space-y-1.5">
                  <h4 className="font-semibold text-foreground text-xs">Research Steps Executed</h4>
                  <div className="space-y-3 pl-1">
                    {message.thinking!.map((step: StepResult, index: number) => (
                      <div key={step.step_index ?? index} className="border-l-2 border-border pl-3 py-1 space-y-1">
                        <div className="flex items-center text-xs font-medium text-foreground">
                          <Search className="h-3 w-3 mr-1.5 flex-shrink-0" />
                          <span>{step.step_name || `Step ${step.step_index + 1}`}</span>
                        </div>
                        {/* Display findings preview */}
                        {step.findings_preview && (
                          <p className="text-muted-foreground text-xs break-words">
                            {step.findings_preview}
                          </p>
                        )}
                        {/* Display sources */} 
                        {step.sources && step.sources.length > 0 && (
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 pt-1">
                            <span className="text-xs font-medium text-foreground">Sources:</span>
                            {step.sources.map((source, srcIndex) => (
                              <a 
                                key={srcIndex} 
                                href={source} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-500 underline truncate max-w-[200px]"
                              >
                                <LinkIcon size={12} />
                                <span>{new URL(source).hostname}</span> {/* Show hostname */} 
                              </a>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    );
  };

  // Helper to render steps during streaming OR in the final view
  const renderStepsContent = (steps: StepResult[]) => {
    if (!steps || steps.length === 0) return null;
    return (
      <div className="space-y-1.5 mt-2"> {/* Add margin top */} 
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
                    {step.sources.map((source, srcIndex) => (
                      <a 
                        key={srcIndex} 
                        href={source} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-500 underline truncate max-w-[200px]"
                      >
                        <LinkIcon size={12} />
                        <span>{new URL(source).hostname}</span> 
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
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
            
            {/* Render intermediate/streaming text (like plan announcement) */}
            {message.text && (
              <div className="prose prose-sm dark:prose-invert max-w-none break-words mb-2">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]} 
                  components={{ a: ({node, ...props}) => <a {...props} target="_blank" rel="noopener noreferrer" /> }}
                >
                  {message.text}
                </ReactMarkdown>
              </div>
            )}

            {/* Render accumulating steps during streaming */} 
            {renderStepsContent(message.thinking || [])} 

          </div>
        ) : (
          // Render final message
          <>
            {/* Final Collapsible Reasoning */} 
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