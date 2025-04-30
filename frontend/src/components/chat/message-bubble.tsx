import React from 'react';
import { Message } from './chat-layout'; // Assuming Message type is exported from ChatLayout
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Bot, User, Search, Eye } from 'lucide-react'; // Icons for sender and search steps
import ReactMarkdown from 'react-markdown'; // For rendering final report
import remarkGfm from 'remark-gfm'; // GF Markdown support

// Define the structure for the step result
interface ExecutedStep {
  step: string;
  result: string | any;
}

interface StepResult {
  executed_steps?: ExecutedStep[];
  step?: string;
  result?: string | any;
  step_index?: number;
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

  // Render the thinking details (research steps)
  const renderThinking = () => {
    if (!message.thinking || message.thinking.length === 0) return null;
    
    return (
      <Accordion type="single" collapsible className="w-full text-xs mb-2 border-b border-zinc-200 dark:border-zinc-700 pb-2">
        <AccordionItem value="thinking" className="border-0">
          <AccordionTrigger className="text-muted-foreground hover:no-underline py-2 px-0 font-normal flex gap-1.5 items-center">
            <Eye size={14} />
            <span>Show reasoning</span>
          </AccordionTrigger>
          <AccordionContent className="pt-2 pb-0 px-0">
            <div className="space-y-3">
              {/* First display plan if available */}
              {message.plan && message.plan.length > 0 && (
                <div className="space-y-1.5">
                  <div className="font-medium text-xs">Research Plan</div>
                  <ol className="list-decimal list-inside text-xs text-muted-foreground space-y-1 pl-2">
                    {message.plan.map((step, index) => (
                      <li key={index}>{step}</li>
                    ))}
                  </ol>
                </div>
              )}
              
              {/* Display all thinking steps as one section */}
              <div className="space-y-1.5">
                <div className="font-medium text-xs">Research Steps</div>
                <div className="space-y-3 pl-2">
                  {message.thinking.map((step: StepResult, index: number) => (
                    <div key={index} className="space-y-1">
                      <div className="flex items-center text-xs font-medium text-muted-foreground">
                        <Search className="h-3 w-3 mr-1" />
                        <span>Step {step.step_index !== undefined ? step.step_index + 1 : index + 1}</span>
                      </div>
                      
                      {/* Handle direct step result format */}
                      {step.step && (
                        <div className="text-xs">
                          <div className="font-medium text-muted-foreground">{step.step}</div>
                          <div className="text-muted-foreground whitespace-pre-wrap text-xs mt-1">
                            {typeof step.result === 'string' 
                              ? step.result.substring(0, 300) + (step.result.length > 300 ? '...' : '')
                              : JSON.stringify(step.result).substring(0, 300) + '...'}
                          </div>
                        </div>
                      )}
                      
                      {/* Handle steps with executed_steps array */}
                      {step.executed_steps && step.executed_steps.map((executedStep: ExecutedStep, stepIdx: number) => (
                        <div key={stepIdx} className="text-xs">
                          <div className="font-medium text-muted-foreground">{executedStep.step}</div>
                          <div className="text-muted-foreground whitespace-pre-wrap text-xs mt-1">
                            {typeof executedStep.result === 'string' 
                              ? executedStep.result.substring(0, 300) + (executedStep.result.length > 300 ? '...' : '')
                              : JSON.stringify(executedStep.result).substring(0, 300) + '...'}
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
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
            
            {/* Display plan during streaming if available */}
            {message.plan && message.plan.length > 0 && (
              <div className="space-y-1.5 mb-2">
                <div className="font-medium text-xs">Research Plan</div>
                <ol className="list-decimal list-inside text-xs text-muted-foreground space-y-1 pl-2">
                  {message.plan.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>
            )}
            
            {/* Display intermediate text if needed */}
            {message.text && <p className="text-xs text-muted-foreground whitespace-pre-wrap break-words">{message.text}</p>} 
          </div>
        ) : (
          // Render final message
          <>
            {/* Show collapsible thinking details */}
            {renderThinking()}
            
            {/* Render markdown content */}
            <div className="prose prose-sm dark:prose-invert max-w-none break-words">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.text}
              </ReactMarkdown>
            </div>
          </>
        )}
      </div>
    </div>
  );
} 