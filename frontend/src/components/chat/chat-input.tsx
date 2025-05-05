'use client'

import React, { useState, useRef, useEffect } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { SendHorizonal, RotateCcw } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (input: string) => void;
  isLoading: boolean;
}

export default function ChatInput({ onSendMessage, isLoading }: ChatInputProps) {
  const [inputText, setInputText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-focus the textarea when the component mounts
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  const handleSend = () => {
    if (inputText.trim() && !isLoading) {
      onSendMessage(inputText);
      setInputText('');
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault(); // Prevent newline on Enter
      handleSend();
    }
  };

  return (
    <div className="max-w-3xl mx-auto w-full">
      <div className="relative flex items-center rounded-lg border border-input bg-background overflow-hidden focus-within:ring-1 focus-within:ring-ring">
        <Textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask your research question..."
          className="min-h-12 resize-none border-0 p-3 pr-14 focus-visible:ring-0 focus-visible:ring-transparent flex-1 transition-height duration-100"
          rows={1}
          disabled={isLoading}
        />
        <div className="absolute right-2 bottom-[10px]">
          <Button 
            onClick={handleSend} 
            disabled={isLoading || !inputText.trim()} 
            size="icon" 
            variant="ghost"
            className="h-8 w-8"
          >
            {isLoading ? (
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent" role="status" />
            ) : (
              <SendHorizonal className="h-4 w-4" />
            )}
            <span className="sr-only">Send message</span>
          </Button>
        </div>
      </div>
      <div className="text-xs text-muted-foreground text-center mt-2">
        DeepSearch scours the web for accurate answers
      </div>
    </div>
  );
} 