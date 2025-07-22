using Microsoft.ML;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Caching.Memory;
using LanguageModelApi.Services;
using LanguageModelApi.Models;
using System.Diagnostics;
using Microsoft.AspNetCore.Mvc.Versioning;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Add API versioning
builder.Services.AddApiVersioning(options =>
{
    options.DefaultApiVersion = new ApiVersion(1, 0);
    options.AssumeDefaultVersionWhenUnspecified = true;
    options.ReportApiVersions = true;
});

// Add ML.NET context as a singleton
builder.Services.AddSingleton<MLContext>(new MLContext(seed: 42));

// Add language model service
builder.Services.AddSingleton<LanguageModelService>();

// Add memory cache
builder.Services.AddMemoryCache();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

// API key middleware
app.Use(async (context, next) =>
{
    // Skip API key check for health endpoint and in development
    if (context.Request.Path.StartsWithSegments("/api/health") || 
        app.Environment.IsDevelopment())
    {
        await next();
        return;
    }
    
    // Check for API key
    if (!context.Request.Headers.TryGetValue("X-API-Key", out var extractedApiKey))
    {
        context.Response.StatusCode = 401; // Unauthorized
        await context.Response.WriteAsJsonAsync(new { error = "API key is missing" });
        return;
    }
    
    var apiKey = builder.Configuration["ApiKey"];
    if (apiKey != null && extractedApiKey != apiKey)
    {
        context.Response.StatusCode = 401; // Unauthorized
        await context.Response.WriteAsJsonAsync(new { error = "Invalid API key" });
        return;
    }
    
    await next();
});

// Define API endpoints
app.MapPost("/api/v1/generate", async (
    [FromBody] GenerateTextRequest request,
    [FromServices] LanguageModelService languageModelService,
    [FromServices] IMemoryCache memoryCache) =>
{
    try
    {
        // Start timing
        var stopwatch = Stopwatch.StartNew();
        
        // Check cache
        string cacheKey = $"generate_{request.Prompt}_{request.MaxLength}_{request.Temperature}_{request.TopP}";
        if (memoryCache.TryGetValue(cacheKey, out GenerateTextResponse cachedResponse))
        {
            return Results.Ok(cachedResponse);
        }
        
        // Generate text
        var result = await languageModelService.GenerateTextAsync(
            request.Prompt, 
            request.MaxLength, 
            request.Temperature,
            request.TopP);
            
        stopwatch.Stop();
        
        // Create response
        var response = new GenerateTextResponse
        {
            GeneratedText = result,
            Prompt = request.Prompt,
            Tokens = request.IncludeTokens ? result.Split(' ') : null,
            Metadata = new GenerationMetadata
            {
                Temperature = request.Temperature,
                TopP = request.TopP,
                MaxLength = request.MaxLength,
                GeneratedLength = result.Length,
                ProcessingTimeMs = stopwatch.Elapsed.TotalMilliseconds
            }
        };
        
        // Cache response for 5 minutes
        memoryCache.Set(cacheKey, response, TimeSpan.FromMinutes(5));
        
        return Results.Ok(response);
    }
    catch (Exception ex)
    {
        return Results.Problem(
            title: "Error generating text",
            detail: ex.Message,
            statusCode: 500);
    }
})
.WithName("GenerateText")
.WithOpenApi();

app.MapGet("/api/health", () =>
{
    return Results.Ok(new { 
        Status = "Healthy", 
        Timestamp = DateTime.UtcNow,
        Version = "1.0.0"
    });
})
.WithName("HealthCheck")
.WithOpenApi();

app.Run();