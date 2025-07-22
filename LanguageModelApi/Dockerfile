FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["LanguageModelApi.csproj", "./"]
RUN dotnet restore "LanguageModelApi.csproj"
COPY . .
WORKDIR "/src"
RUN dotnet build "LanguageModelApi.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "LanguageModelApi.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
# Copy the model files
COPY ["models/", "/app/models/"]
ENTRYPOINT ["dotnet", "LanguageModelApi.dll"]