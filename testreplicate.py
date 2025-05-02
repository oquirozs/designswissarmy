import replicate

output = replicate.run(
  "black-forest-labs/flux-schnell",
  input={"prompt": "an iguana on the beach, pointillism"}
)

# Save the generated image
with open('output.png', 'wb') as f:
    f.write(output[0].read()) # type: ignore

print(f"Image saved as output.png")
