#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

#define AppName "Music Tagging Demo"
#define AppPublisher "Chenglin Song"
#define AppExeName "MusicClassification.exe"

[Setup]
AppId={{F3CB548B-4118-45B0-8E4E-8DA885F5AA0E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/f0rKyrie1rving/music-classification
AppSupportURL=https://github.com/f0rKyrie1rving/music-classification/issues
AppUpdatesURL=https://github.com/f0rKyrie1rving/music-classification/releases
DefaultDirName={localappdata}\Programs\MusicClassification
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist-installer
OutputBaseFilename=MusicClassification-Setup-{#AppVersion}-win64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
SetupLogging=yes
CloseApplications=yes
InfoBeforeFile=INSTALLER_NOTICE.txt
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\MusicClassification\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\LICENSE"; DestDir: "{app}\licenses"; DestName: "PROJECT_LICENSE.txt"; Flags: ignoreversion
Source: "..\..\MODEL_LICENSE.md"; DestDir: "{app}\licenses"; Flags: ignoreversion
Source: "..\..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}\licenses"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The app downloads the pinned model and creates only cache data in this folder.
Type: filesandordirs; Name: "{localappdata}\MusicClassification"
