'use client';

import React, { useEffect, useRef } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import MessageBubble from './message-bubble';
import { Message } from './chat-layout';

interface ChatMessagesProps {
  messages: Message[];
}

export default function ChatMessages({ messages }: ChatMessagesProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="flex-1 h-full overflow-hidden relative">
      <ScrollArea className="h-full py-4 px-4">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="max-w-md text-center">
              <h2 className="text-2xl font-bold mb-2">Welcome to DeepSearch</h2>
              <p className="text-muted-foreground">
                Ask me any question and I'll search the web for you.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-6 pb-8">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={scrollRef} /> {/* Scroll anchor */}
          </div>
        )}
      </ScrollArea>
    </div>
  );
} 