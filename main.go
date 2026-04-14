package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	sdk "github.com/bubustack/bubu-sdk-go"
	sttengram "github.com/bubustack/openai-stt-engram/pkg/engram"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if err := sdk.Start(ctx, sttengram.New()); err != nil {
		log.Fatalf("openai-stt engram failed: %v", err)
	}
}
