; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Fall Alert"
#define MyAppVersion "1.0.26"
#define MyAppPublisher "Doctor Ai"
#define MyAppURL "https://www.ddxrx.net/"
#define MyAppExeName "fall_detector.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{39466A16-7765-4378-B8B6-074CFAC70839}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
OutputDir=D:\DoctorAI\Patient_Fall_Detection\dist\out
OutputBaseFilename="Fall Alert"
SetupIconFile=D:\DoctorAI\Patient_Fall_Detection\icon.ico

;InfoAfterFile=C:\Users\z\Desktop\p\min2\dist\SETUP.txt
; Remove the following line to run in administrative install mode (install for all users.)

;DisableDirPage=yes
;DisableProgramGroupPage=yes
;PrivilegesRequired=lowest

SolidCompression=yes
Compression=lzma2/ultra64
LZMAUseSeparateProcess=yes
LZMADictionarySize=1048576
LZMANumFastBytes=273

;LZMAMatchFinder = BT
;MergeDuplicateFiles = yes

WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}";

[Files]
Source: "D:\DoctorAI\Patient_Fall_Detection\dist\fall_detector\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "D:\DoctorAI\Patient_Fall_Detection\dist\fall_detector\*"; DestDir: "{app}"; Flags: ignoreversion  recursesubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

